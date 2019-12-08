import attr
from functools import partial
import json
import os
import pty
import signal
import socket
import subprocess
import sys
import tempfile
import traceback

import trio
from typing import (
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    NoReturn,
    Optional,
    Sequence,
    Union,
)
from async_exit_stack import AsyncExitStack
from async_generator import asynccontextmanager

from . import rlproxy
from ._streams import TextReceiveStream
from ._multi_cancel import MultiCancelScope
from ._subprocess import run_process


@attr.s(auto_attribs=True)
class InteractiveRequest:
    class Error(Exception):
        pass

    command: str
    send_control: Callable[[Sequence[str]], Awaitable[None]]
    cancel_scope: trio.CancelScope = attr.Factory(trio.CancelScope)

    async def respond(self, message: str) -> None:
        await self.send_control(["output", message])

    def error(self, message: str) -> NoReturn:
        raise InteractiveRequest.Error(message)


@asynccontextmanager
async def handle_interactive_request(
    command: str,
    send_control: Callable[[Sequence[str]], Awaitable[None]],
    cancel_scope: trio.CancelScope,
) -> AsyncIterator[InteractiveRequest]:

    request = InteractiveRequest(command, send_control, cancel_scope)
    try:
        with cancel_scope:
            yield request
    except InteractiveRequest.Error as ex:
        await request.respond(f"error: {ex}\n")
    except BaseException as ex:

        def remove_cancelled(exc: BaseException) -> Optional[BaseException]:
            if isinstance(exc, trio.Cancelled):
                return None
            return exc

        ex = trio.MultiError.filter(remove_cancelled, ex)
        await request.respond(f"*** {ex!r}\n\n")
        await request.respond("".join(traceback.format_exc()) + "\n")
    finally:
        if cancel_scope.cancelled_caught:
            await request.respond("\n")
        await request.send_control(["done"])


@asynccontextmanager
async def interact(
    *,
    prompt: str = "> ",
    greeting: str = "",
    terminal: Optional[trio.abc.Stream] = None,
    use_pty: Optional[bool] = None,
    completer: Callable[[str, str, int], Sequence[str]] = lambda x, y, z: (),
) -> AsyncIterator[trio.abc.ReceiveChannel[AsyncContextManager[InteractiveRequest]]]:
    async def proxy_one_way(source: trio.abc.Stream, sink: trio.abc.Stream) -> None:
        while True:
            chunk = await source.receive_some(4096)
            if not chunk:
                break
            await sink.send_all(chunk)

    async with AsyncExitStack() as stack, trio.open_nursery() as nursery:
        preexec_fn: Optional[Callable[[], None]] = None

        if use_pty is None and terminal is None:
            use_pty = os.isatty(0) or os.isatty(1)

        if use_pty:
            if terminal is not None:
                our_fd, their_fd = pty.openpty()
                our_stream = await stack.enter_async_context(
                    trio.hazmat.FdStream(our_fd)
                )
                nursery.start_soon(proxy_one_way, terminal, our_stream)
                nursery.start_soon(proxy_one_way, our_stream, terminal)
            else:
                try:
                    their_fd = os.open("/dev/tty", os.O_RDWR)
                except OSError:
                    raise RuntimeError(
                        "You must pass a 'terminal' argument when running "
                        "with no controlling terminal and use_pty=True."
                    )
                preexec_fn = os.setpgrp
            stack.callback(os.close, their_fd)
        else:
            our_sock, their_sock = socket.socketpair()
            their_fd = their_sock.detach()
            our_stream = await stack.enter_async_context(
                trio.SocketStream(trio.socket.from_stdlib_socket(our_sock))
            )
            if terminal is None:
                in_ = trio.hazmat.FdStream(os.open("/proc/self/fd/0", os.O_RDONLY))
                out = trio.hazmat.FdStream(os.open("/proc/self/fd/1", os.O_WRONLY))
                terminal = await stack.enter_async_context(trio.StapledStream(out, in_))
            assert terminal is not None
            nursery.start_soon(proxy_one_way, terminal, our_stream)
            nursery.start_soon(proxy_one_way, our_stream, terminal)

        with trio.socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as ssock:
            with tempfile.TemporaryDirectory(prefix="trio-readline.") as tmpdir:
                sockname = os.path.join(tmpdir, "control.sock")
                await ssock.bind(sockname)
                ssock.listen(1)

                process = await nursery.start(  # type: ignore
                    partial(
                        run_process,
                        [sys.executable, "-m", rlproxy.__name__, sockname],
                        stdin=their_fd,
                        stdout=their_fd,
                        preexec_fn=preexec_fn,
                    ),
                    name="<rlproxy process>",
                )

                # Make the readline-proxy process part of the foreground process
                # group so that it gets Ctrl+C rather than us.
                os.tcsetpgrp(their_fd, process.pid)

                sock, _ = await ssock.accept()
                control = await stack.enter_async_context(trio.SocketStream(sock))

        if terminal is None and use_pty:

            @stack.callback
            def restore_pgrp() -> None:
                old_handler = signal.getsignal(signal.SIGTTOU)
                signal.signal(signal.SIGTTOU, signal.SIG_IGN)
                try:
                    # This will raise SIGTTOU if we're not in the
                    # foreground process group, which we have to ignore
                    os.tcsetpgrp(their_fd, os.getpgrp())
                finally:
                    signal.signal(signal.SIGTTOU, old_handler)

        async def send_control(words: Sequence[str]) -> None:
            await control.send_all(json.dumps(words).encode("utf-8") + b"\n")

        send_channel, receive_channel = trio.open_memory_channel[
            AsyncContextManager[InteractiveRequest]
        ](1)
        active_request_scopes = MultiCancelScope()

        async def handle_control() -> None:
            nonlocal active_request_scopes
            try:
                async for line in TextReceiveStream(control, encoding="utf-8"):
                    request = json.loads(line)
                    if request[0] == "complete":
                        await send_control(
                            completer(request[1], request[2], int(request[3]))
                        )
                    elif request[0] == "run":
                        await send_channel.send(
                            handle_interactive_request(
                                request[1],
                                send_control,
                                active_request_scopes.open_child(),
                            )
                        )
                    elif request[0] == "eof":
                        await send_channel.send(
                            handle_interactive_request(
                                "quit", send_control, active_request_scopes.open_child()
                            )
                        )
                    elif request[0] == "interrupt":
                        active_request_scopes.cancel()
                        active_request_scopes = MultiCancelScope()
                    else:
                        raise ValueError(
                            f"readline proxy sent invalid request {request!r}"
                        )
            except trio.BrokenResourceError:
                pass
            finally:
                await send_channel.aclose()

        await send_control(["prompt", prompt])
        if greeting:
            if greeting[-1] != "\n":
                greeting += "\n"
            await send_control(["output", greeting])
        await send_control(["done"])

        nursery.start_soon(handle_control)
        async with receive_channel:
            yield receive_channel
        nursery.cancel_scope.cancel()
