# Very similar to trio.run_process(), but with some extra features

import signal
import subprocess
import trio
from trio_typing import TaskStatus
from typing import Optional, Sequence, Union, Any, List


async def run_process(
    command: Union[str, Sequence[str]],
    *,
    input: Optional[bytes] = None,
    check: bool = True,
    passthrough: bool = False,
    preserve_result: bool = False,
    shutdown_signal: signal.Signals = signal.SIGTERM,
    shutdown_timeout: float = 5,
    task_status: TaskStatus[trio.Process] = trio.TASK_STATUS_IGNORED,
    **options: Any,
) -> subprocess.CompletedProcess:
    """Run ``command`` in a subprocess, wait for it to complete, and
    return a :class:`subprocess.CompletedProcess` instance describing
    the results.

    If cancelled, :func:`run_process` terminates the subprocess and
    waits for it to exit before propagating the cancellation, like
    :meth:`Process.aclose`.  If you need to allow the process to
    perform an orderly shutdown instead of being forcibly terminated,
    see the ``shutdown_signal`` and ``shutdown_timeout`` arguments.
    If you need to be able to tell what partial output the process
    produced before a timeout, see the ``preserve_result`` argument.

    The default behavior of :func:`run_process` is designed to isolate
    the subprocess from potential impacts on the parent Trio process, and to
    reduce opportunities for errors to pass silently. Specifically:

    * The subprocess's standard input stream is set up to receive the
      bytes provided as ``input``.  Once the given input has been
      fully delivered, or if none is provided, the subprocess will
      receive end-of-file when reading from its standard input.

    * The subprocess's standard output and standard error streams are
      individually captured and returned as bytestrings from
      :func:`run_process`.

    * If the subprocess exits with a nonzero status code, indicating failure,
      :func:`run_process` raises a :exc:`subprocess.CalledProcessError`
      exception rather than returning normally. The captured outputs
      are still available as the ``stdout`` and ``stderr`` attributes
      of that exception.

    To suppress the :exc:`~subprocess.CalledProcessError` on failure,
    pass ``check=False``. To run the subprocess without I/O capturing,
    pass ``passthrough=True``. To redirect some standard streams
    differently than others, use the lower-level ``stdin``,
    ``stdout``, and/or ``stderr`` :ref:`options <subprocess-options>`.

    If you specify ``passthrough=True`` or a value for ``stdin`` other
    than ``PIPE``, you can't specify ``input`` (because we'd have no
    way to send it). If you specify ``passthrough=True`` or a value
    for ``stdout`` or ``stderr`` other than ``PIPE``, you can't
    observe the subprocess's output or errors; the corresponding
    attributes of the returned returned
    :class:`~subprocess.CompletedProcess` object will be ``None``.

    Args:
      command (list or str): The command to run. Typically this is a
          sequence of strings such as ``['ls', '-l', 'directory with spaces']``,
          where the first element names the executable to invoke and the other
          elements specify its arguments. With ``shell=True`` in the
          ``**options``, or on Windows, ``command`` may alternatively
          be a string, which will be parsed following platform-dependent
          quoting rules.
      input (bytes): The input to provide to the subprocess on its
          standard input stream. If you want the subprocess's input
          to come from something other than data specified at the time
          of the :func:`run_process` call, you can specify a redirection
          using the lower-level ``stdin`` option; then ``input`` must
          be unspecified or None.
      check (bool): If false, don't validate that the subprocess exits
          successfully. You should be sure to check the
          ``returncode`` attribute of the returned object if you pass
          ``check=False``, so that errors don't pass silently.
      passthrough (bool): If true, set up the subprocess to inherit the
          parent Trio process's standard streams; for example, if the parent
          Trio process is running in an interactive console, the subprocess
          will be able to interact with the user via that console. Only
          one call to :func:`run_process` should be active at a time with
          ``passthrough=True``, to avoid different processes' I/O being
          unpredictably interleaved.
      preserve_result (bool): If true, return normally even if cancelled,
          in order to give the caller a chance to inspect the process's
          partial output before a timeout.
      shutdown_signal (int): The signal that the process will initially be
          sent when :func:`run_process` is cancelled.
      shutdown_timeout (float): The number of seconds we will wait for the
          process to exit in response to the ``shutdown_signal``, before
          forcibly killing it with ``SIGKILL``.
      task_status: This function can be used with ``nursery.start``.
          If it is, it returns the :class:`Process` object, so that other tasks
          can send signals to the subprocess or wait for it to exit.
          (They shouldn't try to send or receive on the subprocess's
          input and output streams, because :func:`run_process` is already
          doing that.)
      **options: :func:`run_process` also accepts any :ref:`general subprocess
          options <subprocess-options>` and passes them on to the
          :class:`~trio.Process` constructor.

    Returns:
      A :class:`subprocess.CompletedProcess` instance describing the
      return code and outputs.

    Raises:
      subprocess.CalledProcessError: if ``check=False`` is not passed
          and the process exits with a nonzero exit status
      OSError: if an error is encountered starting or communicating with
          the process

    .. note:: The child process runs in the same process group as the parent
       Trio process, so a Ctrl+C will be delivered simultaneously to both
       parent and child. If you don't want this behavior, consult your
       platform's documentation for starting child processes in a different
       process group.

    .. warning::
       If you pass ``preserve_result=True``, :func:`run_process` has
       no way to directly propagate a :exc:`~trio.Cancelled`
       exception, so you should execute a checkpoint immediately after
       :func:`run_process` returns in order to propagate the cancellation::

           with trio.move_on_after(1) as scope:
               result = await trio.run_process(
                   "echo -n test; sleep 10", shell=True, preserve_result=True
               )
               await trio.sleep(0)   # <-- like so

    """
    default_redirect = None if passthrough else subprocess.PIPE
    options.setdefault("stdin", default_redirect)
    options.setdefault("stdout", default_redirect)
    options.setdefault("stderr", default_redirect)

    if input is not None and options["stdin"] != subprocess.PIPE:
        raise ValueError("can't provide input to a process whose stdin is redirected")

    if options["stdin"] == subprocess.PIPE and not input:
        options["stdin"] = subprocess.DEVNULL

    stdout_chunks: List[bytes] = []
    stderr_chunks: List[bytes] = []

    try:
        async with await trio.open_process(command, **options) as proc:
            task_status.started(proc)

            async def feed_input() -> None:
                if proc.stdin is not None:
                    async with proc.stdin:
                        try:
                            if input:
                                await proc.stdin.send_all(input)
                        except trio.BrokenResourceError:
                            pass

            async def read_output(
                stream: Optional[trio.abc.ReceiveStream], chunks: List[bytes]
            ) -> None:
                if stream is not None:
                    async with stream:
                        while True:
                            chunk = await stream.receive_some(32768)
                            if not chunk:
                                break
                            chunks.append(chunk)

            async def handle_io(shield_scope: trio.CancelScope) -> None:
                with shield_scope:
                    async with trio.open_nursery() as nursery:
                        nursery.start_soon(feed_input)
                        nursery.start_soon(read_output, proc.stdout, stdout_chunks)
                        nursery.start_soon(read_output, proc.stderr, stderr_chunks)

            async with trio.open_nursery() as nursery:
                shield_scope = trio.CancelScope(shield=True)
                nursery.start_soon(
                    handle_io, shield_scope, name="<subprocess {!r}>".format(command)
                )
                try:
                    await proc.wait()
                finally:
                    if proc.returncode is None and shutdown_timeout > 0:
                        # still running -- wait() was cancelled
                        proc.send_signal(shutdown_signal)
                        with trio.move_on_after(shutdown_timeout) as scope:
                            scope.shield = True
                            await proc.wait()
                        # if it keeps running after the shutdown_timeout,
                        # Process.aclose() will SIGKILL it

                    # Let the process exit before we cut off its I/O
                    # capturing, so that we capture its output due to the
                    # graceful shutdown signal if any
                    shield_scope.shield = False

    except trio.Cancelled:
        if preserve_result:
            check = False  # don't complain that we killed the process
            # swallow the cancellation
        else:
            raise

    stdout = b"".join(stdout_chunks) if proc.stdout is not None else None
    stderr = b"".join(stderr_chunks) if proc.stderr is not None else None

    assert proc.returncode is not None
    if proc.returncode and check:
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=stdout, stderr=stderr
        )
    else:
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
