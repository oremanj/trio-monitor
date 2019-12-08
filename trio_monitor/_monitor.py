import attr
import codeop
import dis
import enum
import inspect
import itertools
import outcome
import os
import signal
import sys
import traceback
import trio
import trio_typing
import types
import weakref
from contextlib import closing, contextmanager
from functools import partial
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterator,
    List,
    NewType,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)
from types import CodeType, FrameType

from ._interact import interact, InteractiveRequest


_Handler = TypeVar("_Handler", bound=Callable[..., Awaitable[None]])


"""
@attr.s(auto_attribs=True)
class Interceptor:
    task: trio.hazmat.Task
    resume_with: Optional[outcome.Outcome]
    original_coro:
"""

RestOfLine = NewType("RestOfLine", str)


@attr.s(auto_attribs=True)
class AsyncREPL:
    namespace: Dict[str, object] = attr.Factory(dict)
    pending: List[str] = attr.Factory(list)
    compiler: codeop.CommandCompiler = attr.Factory(codeop.CommandCompiler)

    def push(self, incoming_line: str) -> Optional[CodeType]:
        self.pending.append(incoming_line)
        sync_input = "\n".join(self.pending)
        try:
            try:
                if self.pending[-1] != "" and self.compiler(sync_input) is None:
                    return None
                try:
                    code = compile(sync_input, "<stdin>", "eval")
                except SyntaxError:
                    code = compile(sync_input, "<stdin>", "exec")
            except SyntaxError:
                if not (set(sync_input.split()) & {"async", "await"}):
                    raise

                # See if we can make the error go away by wrapping this
                # code in an 'async def'. Try first as an expression, then
                # as a statement.
                async_expr_input = (
                    "async def _trio_repl_async_expr():\n"
                    + "\n".join(
                        "  " + ("return " if idx == 0 else "") + line
                        for idx, line in enumerate(self.pending)
                    )
                    + "\n"
                )
                try:
                    code = self.compiler(async_expr_input)
                except SyntaxError:
                    async_stmt_input = "async def _trio_repl_async():\n" + "\n".join(
                        "  " + line for line in self.pending
                    )
                    if len(self.pending) == 1:
                        async_stmt_input += "\n"
                    if (
                        self.pending[-1] != ""
                        and self.compiler(async_stmt_input) is None
                    ):
                        return None
                    else:
                        async_stmt_input += "\n  return locals()\n"
                        code = compile(async_stmt_input, "<stdin>", "exec")
        except Exception:
            self.pending = []
            raise

        if code is not None:
            self.pending = []
        return code  # type: ignore


@attr.s(auto_attribs=True)
class Dispatcher:
    description: str
    prompt: str
    parent: Optional["Dispatcher"] = None
    exit_to: Optional["Dispatcher"] = None
    # {command: (handler, (usage_text, help_text))}
    # {alias: (handler, None)}
    table: Dict[
        str, Tuple[Callable[..., Awaitable[None]], Optional[Tuple[str, str]]]
    ] = attr.Factory(dict)

    @staticmethod
    def is_simple_type(ty: object) -> bool:
        if ty in (str, int, float, bool, RestOfLine):
            return True
        if isinstance(ty, type) and issubclass(ty, enum.Enum):
            return True
        if type(ty).__name__ == "_Union":
            return all(
                arm is type(None) or Dispatcher.is_simple_type(arm)
                for arm in ty.__args__  # type: ignore
            )
        return False

    @staticmethod
    def usage_string(name: str, fn: _Handler) -> str:
        usage_words = [name]
        type_hints = get_type_hints(fn)
        saw_rest = False
        for param in list(inspect.signature(fn).parameters.values())[2:]:
            if param.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            ):
                # these are for internal use, not filled in from the command line
                continue
            if param.name not in type_hints:
                raise RuntimeError(
                    f"{fn!r} ({name}) parameter {param.name!r} must have a "
                    "type hint so we know how to parse it"
                )
            if type_hints[param.name] is RestOfLine:
                saw_rest = True
            elif saw_rest:
                raise RuntimeError(
                    f"{fn!r} ({name}): RestOfLine can only be used for the "
                    f"last parameter"
                )
            cooked_name = param.name.replace("_", "-")
            if not Dispatcher.is_simple_type(type_hints[param.name]):
                cooked_name += "-id"

            if saw_rest or param.kind == inspect.Parameter.VAR_POSITIONAL:
                dots = "..."
            else:
                dots = ""
            if param.default is not inspect.Parameter.empty:
                usage_words.append(f"[{cooked_name}{dots}]")
            else:
                usage_words.append(f"<{cooked_name}{dots}>")
        return " ".join(usage_words)

    @staticmethod
    def parse_simple_arg(param_type: object, value: str) -> object:
        if param_type in (str, int, float, RestOfLine):
            if param_type is RestOfLine:
                param_type = str
            assert isinstance(param_type, type)
            try:
                return param_type(value)  # type: ignore
            except ValueError:
                raise ValueError(f"got {value!r} where {param_type.__name__} expected")
        if param_type is bool:
            if value.lower() in ("0", "f", "false", "n", "no"):
                return False
            if value.lower() in ("1", "t", "true", "y", "yes"):
                return True
            raise ValueError(f"got {value!r} where boolean expected")
        if isinstance(param_type, type) and issubclass(param_type, enum.Enum):
            try:
                if value.isdigit():
                    return param_type(int(value))  # type: ignore
                else:
                    return param_type[value]  # type: ignore
            except (KeyError, ValueError):
                raise ValueError(
                    f"got {value!r} where valid {param_type.__name__} "
                    "expected (as integer or string)"
                )
        raise RuntimeError(f"parse_simple_arg() can't handle a {param_type}")

    def add(self, *names: str, help: str = "") -> Callable[[_Handler], _Handler]:
        def decorate(fn: _Handler) -> _Handler:
            cooked_help = help or f"undocumented {fn}"
            if len(names) > 1:
                descr = "aliases" if len(names) > 2 else "alias"
                cooked_help += f"  [{descr}: {', '.join(names[1:])}]"

            for idx, name in enumerate(names):
                if name in self.table:
                    raise ValueError(
                        f"{self.table[name]!r} and {fn!r} can't both handle {name!r}"
                    )
                self.table[name] = (
                    fn,
                    None if idx != 0 else (self.usage_string(name, fn), cooked_help),
                )
            return fn

        return decorate

    def names(self) -> List[str]:
        ret = self.parent.names() if self.parent is not None else []
        ret.extend(self.table.keys())
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self.table or (self.parent is not None and name in self.parent)

    def __getitem__(self, name: str) -> Callable[..., Awaitable[None]]:
        try:
            return self.table[name][0]
        except KeyError:
            if self.parent is not None:
                try:
                    return self.parent[name]
                except KeyError:
                    pass
            raise

    async def do_help(
        self,
        monitor: "TrioMonitor",
        request: InteractiveRequest,
        *,
        suppress: FrozenSet[str] = frozenset(),
    ) -> None:
        if self.parent is not None:
            await self.parent.do_help(
                monitor, request, suppress=suppress | self.table.keys()
            )
        await request.respond(f"--- {self.description} commands ---\n")
        entries = [
            (usage_and_help[0].replace("_default ", ""), usage_and_help[1])
            for name, (_, usage_and_help) in self.table.items()
            if name not in suppress and usage_and_help is not None
        ]
        usage_width = max(len(usage) for usage, _ in entries) + 3
        for usage_text, help_text in entries:
            usage_and_dots = (usage_text + " ").ljust(usage_width, ".")
            await request.respond(f"{usage_and_dots} {help_text}\n")
        await request.respond("\n")

    async def do_quit(
        self, monitor: "TrioMonitor", request: InteractiveRequest
    ) -> None:
        if self.parent is None:
            await request.send_control(["quit"])
            return
        else:
            monitor.dispatcher = self.exit_to or self.parent
            await request.send_control(["prompt", monitor.dispatcher.prompt])

    def __attrs_post_init__(self) -> None:
        self.add("help", help="show a list of valid commands")(self.do_help)
        self.add(
            "quit",
            "exit",
            help=(
                f"exit to {self.parent.description}"
                if self.parent is not None
                else "exit the monitor"
            ),
        )(self.do_quit)


# We need the name TrioMonitor to exist when Dispatcher.add() calls
# typing.get_type_hints()
TrioMonitor = None


@attr.s(auto_attribs=True)  # type: ignore  # (the "redefinition")
class TrioMonitor:
    main_dispatcher: ClassVar[Dispatcher] = Dispatcher("top level", "trio> ")
    repl_dispatcher: ClassVar[Dispatcher] = Dispatcher("REPL", ">>> ", main_dispatcher)
    inspect_dispatcher: ClassVar[Dispatcher] = Dispatcher(
        "inspector", "trio(...)> ", main_dispatcher
    )
    debug_dispatcher: ClassVar[Dispatcher] = Dispatcher(
        "debugger", "trio(...)> ", inspect_dispatcher, main_dispatcher
    )

    dispatcher: Dispatcher = main_dispatcher
    restart: bool = False
    obj_from_id: "weakref.WeakValueDictionary[int, object]" = attr.Factory(
        weakref.WeakValueDictionary
    )
    id_from_obj: "weakref.WeakKeyDictionary[object, int]" = attr.Factory(
        weakref.WeakKeyDictionary
    )
    task_id_from_frame_owner: "weakref.WeakKeyDictionary[object, int]" = attr.Factory(
        weakref.WeakKeyDictionary
    )

    id_generator: Iterator[int] = attr.Factory(itertools.count)

    def close(self) -> None:
        pass

    _lookup_weird_type: ClassVar[Dict[object, str]] = {
        trio.CancelScope: "get_cancel_scope",
        trio_typing.Nursery: "get_nursery",
        trio.hazmat.Task: "get_task",
        Tuple[trio.hazmat.Task, FrameType]: "get_frame",
        FrameType: "get_frame_only",
    }

    def parse_arg(self, param_type: object, name: str, value: str) -> object:
        if type(param_type).__name__ == "_Union":
            for arm in param_type.__args__:  # type: ignore
                if arm is type(None):
                    continue
                try:
                    return self.parse_arg(arm, name, value)
                except ValueError:
                    pass
            raise ValueError("couldn't parse as any of {param_type.__arguments__!r}")

        elif Dispatcher.is_simple_type(param_type):
            try:
                return Dispatcher.parse_simple_arg(param_type, value)
            except ValueError as ex:
                raise ValueError(f"invalid value for {name!r}: {ex}")

        elif param_type in self._lookup_weird_type:
            try:
                ident = int(value)
            except ValueError:
                raise ValueError(
                    f"invalid value for {name!r}: need an integer ID that was "
                    f"previously listed on the left-hand side of the output "
                    f"from the 'tree' command"
                )
            try:
                lookup_attr = self._lookup_weird_type[param_type]
                return getattr(self, lookup_attr)(ident)  # type: ignore
            except ValueError as ex:
                raise ValueError(f"invalid value for {name!r}: {ex}")

        else:
            raise ValueError(
                f"internal error: don't know how to produce the "
                f"{param_type!r} that {name!r} needs"
            )

    async def handle(self, request: InteractiveRequest) -> None:
        if not request.command.strip() and "_default" not in self.dispatcher:
            return
        command, _, rest = request.command.partition(" ")
        try:
            handler = self.dispatcher[command]
        except KeyError:
            try:
                handler = self.dispatcher["_default"]
            except KeyError:
                request.error(
                    f"invalid command {command!r}; I know about "
                    + repr(self.dispatcher.names())
                )
            else:
                rest = request.command

        args: List[object] = []
        params = [
            param
            for param in inspect.signature(handler).parameters.values()
            if param.kind
            not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
        ][2:]
        type_hints = get_type_hints(handler)
        maxsplit = -1
        for idx, param in enumerate(params):
            if type_hints[param.name] is RestOfLine:
                maxsplit = idx
                break
        if maxsplit == 0:
            words = [rest]
        else:
            words = rest.split(maxsplit=maxsplit)
        for idx, param in enumerate(params):
            if idx >= len(words):
                if param.default is not inspect.Parameter.empty:
                    continue
                else:
                    request.error(f"need a value for {param.name!r}")
            param_type = type_hints[param.name]
            try:
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    args.extend(
                        self.parse_arg(param_type, param.name, words[jdx])
                        for jdx in range(idx, len(words))
                    )
                    break
                else:
                    args.append(self.parse_arg(param_type, param.name, words[idx]))
            except ValueError as ex:
                request.error(str(ex))
        await handler(self, request, *args)

    def get_by_id(self, ident: int) -> object:
        try:
            return self.obj_from_id[ident]
        except KeyError:
            raise ValueError(f"no object with id {ident} (maybe it's been destroyed?)")

    def get_cancel_scope(self, ident: int) -> trio.CancelScope:
        obj = self.get_by_id(ident)
        if isinstance(obj, trio.CancelScope):
            return obj
        elif isinstance(obj, trio_typing.Nursery):
            return obj.cancel_scope
        else:
            raise ValueError(
                f"I need a cancel scope or nursery, not {type(obj).__name__}"
            )

    def get_nursery(self, ident: int) -> trio_typing.Nursery:
        obj = self.get_by_id(ident)
        if isinstance(obj, trio_typing.Nursery):
            return obj
        else:
            raise ValueError(f"I need a nursery, not {type(obj).__name__}")

    def get_task(self, ident: int) -> trio.hazmat.Task:
        obj = self.get_by_id(ident)
        if isinstance(obj, trio.hazmat.Task):
            return obj
        else:
            raise ValueError(f"I need a task, not {type(obj).__name__}")

    def get_frame(self, ident: int) -> Tuple[trio.hazmat.Task, FrameType]:
        from . import _inspect

        obj = self.get_by_id(ident)
        if isinstance(obj, trio.hazmat.Task):
            try:
                return obj, _inspect.task_traceback(obj)[-1][1]
            except IndexError:
                raise ValueError("sorry, you can't debug the monitor task")
        try:
            task = self.get_task(self.task_id_from_frame_owner[obj])
        except KeyError:
            raise ValueError(f"I need a frame or task, not {type(obj).__name__}")

        frame, _ = _inspect.frame_and_next(cast(Awaitable[Any], obj))
        if frame is None:
            raise ValueError("sorry, I don't know how to debug a {type(obj).__name__}")
        return task, frame

    def get_frame_only(self, ident: int) -> FrameType:
        return self.get_frame(ident)[1]

    def id_for(self, obj: object) -> Optional[int]:
        try:
            weakref.ref(obj)
        except TypeError:
            return None
        try:
            return self.id_from_obj[obj]
        except KeyError:
            ident = next(self.id_generator)
            self.id_from_obj[obj] = ident
            self.obj_from_id[ident] = obj
            return ident

    def id_for_frame(self, task: trio.hazmat.Task, owner: object) -> Optional[int]:
        ident = self.id_for(owner)
        if ident is not None and owner not in self.task_id_from_frame_owner:
            task_ident = self.id_for(task)
            assert task_ident is not None
            self.task_id_from_frame_owner[owner] = task_ident
        return ident

    @main_dispatcher.add("tree", help="print task tree [rooted at root-task-id]")
    async def do_tree(
        self, request: InteractiveRequest, root_task: Optional[trio.hazmat.Task] = None
    ) -> None:
        from . import _inspect

        await request.respond("\n")
        for line, obj in _inspect.format_task_tree(root_task, color=True):
            if isinstance(obj, tuple):
                ident = self.id_for_frame(*obj)
            else:
                ident = self.id_for(obj)

            if ident is not None:
                ident_str = f"[{ident:>5}] "
            elif line.strip() != "":
                ident_str = "[     ] "
            else:
                ident_str = "        "

            await request.respond(f"{ident_str}{line}\n")

    @main_dispatcher.add("where", "w", help="print the current backtrace for <task-id>")
    async def do_where(
        self, request: InteractiveRequest, task: trio.hazmat.Task
    ) -> None:
        from . import _inspect

        await request.respond(
            f"        Current stack of {task.name!r} (most recent call last):\n"
        )
        if not task.coro.cr_running:
            frames = _inspect.task_traceback(task)
        else:
            frames = [
                (None, frame)
                for frame, *_ in reversed(inspect.getouterframes(task.coro.cr_frame))
            ]

        for frame_owner, frame in frames:
            ident = self.id_for_frame(task, frame_owner)
            summary, = traceback.format_list(traceback.extract_stack(frame, limit=1))
            for line in summary.splitlines(True):
                if ident is not None:
                    ident_str = f"[{ident:>5}] "
                    ident = None
                else:
                    ident_str = "[     ] "
                await request.respond(f"{ident_str}{line}")
        await request.respond("\n")

    @main_dispatcher.add("vars", help="show local variables in a frame")
    async def do_vars(self, request: InteractiveRequest, frame: FrameType) -> None:
        entries = sorted((name, repr(value)) for name, value in frame.f_locals.items())
        namelen = max(len(name) for name, value in entries) + 2
        for name, value in entries:
            await request.respond(f"{name:>{namelen}} = {value}\n")

    @main_dispatcher.add("cvars", help="show context variables in a task")
    async def do_cvars(
        self, request: InteractiveRequest, task: trio.hazmat.Task
    ) -> None:
        entries = sorted(
            (cvar.name, repr(value)) for cvar, value in task.context.items()
        )
        namelen = max(len(name) for name, value in entries) + 2
        for name, value in entries:
            await request.respond(f"{name:>{namelen}} = {value}\n")

    @main_dispatcher.add("repl", help="start an interactive python interpreter")
    async def do_repl(self, request: InteractiveRequest) -> None:
        self.dispatcher = self.repl_dispatcher
        self.repl = AsyncREPL()
        self.repl.namespace["trio"] = trio
        self.repl.namespace["monitor"] = self
        await request.send_control(["prompt", self.dispatcher.prompt])

    @repl_dispatcher.add("_default", "py", help="execute a python statement")
    async def do_repl_run(
        self, request: InteractiveRequest, statement: RestOfLine
    ) -> None:
        try:
            code = self.repl.push(statement)
        except Exception:
            await request.send_control(["prompt", ">>> "])
            raise

        if code is None:
            await request.send_control(["prompt", "... "])
            return
        await request.send_control(["prompt", ">>> "])
        result = eval(code, self.repl.namespace, self.repl.namespace)
        if result is not None:
            self.repl.namespace["_"] = result
            await request.respond(f"{result!r}\n")
            return

        afn = self.repl.namespace.pop("_trio_repl_async_expr", None)
        if afn is not None:
            afn_is_expr = True
        else:
            afn = self.repl.namespace.pop("_trio_repl_async", None)
            afn_is_expr = False

        if afn is None:
            return

        # Create a new code object which turns all locals into arguments
        co = afn.__code__  # type: ignore
        if len(co.co_varnames) >= 256:
            request.error(
                "sorry, can't handle top-level async expressions with "
                "more than 256 variables"
            )
        args = []
        new_code = bytearray()
        for idx, name in enumerate(co.co_varnames):
            if name in self.repl.namespace:
                args.append(self.repl.namespace[name])
            else:
                # Locals that aren't defined in this function will still
                # get passed as arguments (since the args all have to
                # be adjacent in local-index-space and renumbering the
                # accesses would be too annoying). We pass None, which is
                # not the same as nonexistent; so we need to inject a
                # 'del' statement at the start of the function to get
                # the state back to nonexistent.
                args.append(None)
                new_code.extend((dis.opmap["DELETE_FAST"], idx))

        if new_code:
            # If we added DELETE_FAST opcodes, we need to adjust all
            # absolute jump targets by the amount of code we added.
            adjust = len(new_code)
            for idx in range(0, len(co.co_code), 2):
                op, arg = co.co_code[idx : idx + 2]
                if op in dis.hasjabs:
                    arg += adjust
                    carry = arg >> 8
                    arg &= 0xFF
                    backtrack = 0

                    # If this made the argument go over 256, we need
                    # to carry into an EXTENDED_ARG prefix.
                    while carry:
                        if (
                            len(new_code) >= backtrack + 2
                            and new_code[-backtrack - 2] == dis.opmap["EXTENDED_ARG"]
                        ):
                            extarg = new_code[-backtrack - 1] + carry
                            carry = extarg >> 8
                            extarg &= 0xFF
                            new_code[-backtrack - 1] = extarg
                            backtrack += 2
                            continue
                        # No more EXTENDED_ARG prefixes; add one for
                        # whatever is left.
                        new_code[-backtrack:-backtrack] = (
                            dis.opmap["EXTENDED_ARG"],
                            carry,
                        )
                        break
                new_code.extend((op, arg))

            # And update the line number table.
            new_lnotab = bytearray()
            while adjust > 255:
                new_lnotab.extend((255, 0))
                adjust -= 255
            if adjust:
                new_lnotab.extend((adjust, 0))
            new_lnotab += co.co_lnotab
        else:
            adjust = 0
            new_code = bytearray(co.co_code)
            new_lnotab = bytearray(co.co_lnotab)

        replacement_code = CodeType(
            co.co_nlocals,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co.co_flags,
            bytes(new_code),
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            bytes(new_lnotab),
            co.co_freevars,
            co.co_cellvars,
        )
        afn.__code__ = replacement_code  # type: ignore
        coro = afn(*args)  # type: ignore
        result = await coro
        if afn_is_expr:
            if result is not None:
                self.repl.namespace["_"] = result
                await request.respond(f"{result!r}\n")
            return
        # otherwise the result is the new locals dictionary; update our namespace
        for name in co.co_varnames:
            if name in self.repl.namespace and name not in result:
                del self.repl.namespace[name]
            if name in result:
                self.repl.namespace[name] = result[name]

    @repl_dispatcher.add(
        "snarf", help="copy locals from the given frame to the interactive namespace"
    )
    async def do_repl_snarf(
        self, request: InteractiveRequest, frame: FrameType
    ) -> None:
        self.repl.namespace.update(frame.f_locals)
        await request.respond(
            "snarfed: " + ", ".join(sorted(frame.f_locals.keys())) + "\n"
        )

    @main_dispatcher.add("cancel", help="cancel a nursery or cancel scope")
    async def do_cancel(
        self, request: InteractiveRequest, scope: trio.CancelScope
    ) -> None:
        scope.cancel()
        await request.respond(f"{scope!r}\n")

    @main_dispatcher.add("kill", help="force a task to exit")
    async def do_kill(
        self, request: InteractiveRequest, task: trio.hazmat.Task
    ) -> None:
        class MonitorKill(BaseException):
            pass

        result: Optional[BaseException] = None
        done = trio.Event()
        delivered = False

        @types.coroutine
        def interceptor(underlying_coro):  # type: ignore
            nonlocal result, delivered
            try:
                next_trap = "hello"
                while True:
                    trap_result = yield next_trap
                    if isinstance(trap_result, outcome.Error):
                        if isinstance(trap_result.error, MonitorKill):
                            delivered = True
                    next_trap = underlying_coro.send(trap_result)
                    if not delivered:
                        next_trap = underlying_coro.send(outcome.Error(MonitorKill()))
                        delivered = True
            except StopIteration as ex:
                return ex.value
            except MonitorKill as ex:
                result = ex
            except BaseException as ex:
                result = ex
                raise
            finally:
                done.set()

        async def wrap_interceptor(underlying_coro):  # type: ignore
            return await interceptor(underlying_coro)  # type: ignore

        task.coro = wrap_interceptor(task.coro)  # type: ignore
        assert task.coro.send(None) == "hello"  # type: ignore
        if task._next_send is not None:  # type: ignore
            # Already scheduled -- we'll get it at the next checkpoint.
            pass
        else:
            # Sleeping -- try to abort with a MonitorKill exception.
            def raise_cancel() -> NoReturn:
                raise MonitorKill

            task._attempt_abort(raise_cancel)  # type: ignore

        with trio.move_on_after(0.5) as scope:
            await done.wait()
        if scope.cancelled_caught:
            if delivered:
                await request.respond(
                    "exception delivered; waiting for task to exit...\n"
                )
            else:
                await request.respond("still waiting for task to wake up...\n")
            await done.wait()

        if result is None:
            await request.respond("task exited normally\n")
            return

        if isinstance(result, MonitorKill):
            await request.respond("task exited due to your request:\n\n")
        else:
            await request.respond("task exited due to a different exception:\n\n")
        await request.respond(
            "".join(
                traceback.format_exception(type(result), result, result.__traceback__)
            )
        )

    @main_dispatcher.add(
        "signal", help="send <sig> (name or number) to this Trio process"
    )
    async def do_signal(self, request: InteractiveRequest, sig: signal.Signals) -> None:
        if os.name != "posix":
            request.error("signals are only supported on POSIX")
        os.kill(os.getpid(), sig)
        await request.respond(f"signal {sig} sent\n")

    # more stuff we could support:
    #
    # support running the monitor without readline, e.g. via telnet or
    # programmatically or via repl commands
    #
    # freeze: stop all (non-system) tasks, or all tasks in a nursery, or one task
    # thaw: resume them
    # inspect: "debug" a task without stopping it
    # debug: "debug" a task and keep it stopped while we're debugging
    #   (same as freeze + inspect + thaw when quitting debug mode)
    # mon/monitor: install a tracing instrument and print what it says until ^C
    # catch: wait for [any task | some specific task] to exit with an exception,
    #   then start a post-mortem pdb
    #
    # inspect mode:
    # u/up, d/down, bt/where/w/backtrace, l/list, p/print
    #
    # debug mode: inspect mode plus
    # p/print <expr>, await <expr>, <expr> -- run stuff
    # b/break, tbreak, disable, enable, ignore, condition --
    #   can break on the entry to an async function only!
    # s/step -- step by one checkpoint
    # n/next -- step _by checkpoints_ until not on the same line
    # watch: instrument something to write to the monitoring queue
    #   the contents of which get printed after a c/continue
    # c/continue -- and then Ctrl+C to stop again

    @main_dispatcher.add("reload", help="reload trio monitor module")
    async def do_reload(self, request: InteractiveRequest) -> None:
        import importlib
        from . import _monitor, _inspect

        importlib.reload(_monitor)
        importlib.reload(_inspect)
        await request.respond("reloading\n\n")
        self.restart = True
        await request.send_control(["quit"])


async def run_monitor(
    terminal: Optional[trio.abc.Stream] = None, use_pty: Optional[bool] = None
) -> None:
    try:
        while True:
            with closing(TrioMonitor()) as monitor:
                async with interact(
                    greeting=(
                        "Welcome to the Trio monitor. Type 'help' for a list of "
                        "commands.\n\n"
                    ),
                    prompt="trio> ",
                    terminal=terminal,
                    use_pty=use_pty,
                ) as requests:
                    async for request_manager in requests:
                        async with request_manager as request:
                            await monitor.handle(request)
            if not monitor.restart:
                break
    except Exception as ex:
        sys.stderr.write(
            f"*** unhandled exception in trio monitor: {ex!r}\n\n"
            + "".join(traceback.format_exc())
            + "\n\n"
        )
    finally:
        if terminal is not None:
            await terminal.aclose()


def start_monitor() -> None:
    trio.hazmat.spawn_system_task(run_monitor)


def serve_monitor(port: int) -> None:
    trio.hazmat.spawn_system_task(
        partial(trio.serve_tcp, run_monitor, port, host="127.0.0.1")
    )
