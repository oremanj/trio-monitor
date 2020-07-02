import attr
import collections
import ctypes
import dis
import gc
import inspect


def coroutine_traceback(coro: Coroutine[Any, Any, Any]) -> ...:
    pass


def task_traceback(task: trio.lowlevel.Task) -> List[Tuple[Awaitable[Any], FrameType]]:
    """Return the traceback of a Trio task.

    The result is a list of (awaitable, frame) tuples, with the
    outermost call (for the original async function passed to
    :func:`trio.run` or ``nursery.start_soon()``) first.  The
    awaitable will usually be a coroutine object, but may also be a
    generator-based coroutine, async generator asend/athrow, etc.
    """

    if task.coro.cr_running:
        # The technique we're using doesn't work for frames that are
        # not suspended; we could extract the frame stack in the
        # normal way (sys._getframe) but we wouldn't be able to
        # collect the corresponding awaitables so we keep things
        # consistent by just not bothering.  Sorry, you can't debug
        # your debugger. :-)
        return []

    gen: Optional[Awaitable[Any]] = task.coro
    frames = []
    while gen is not None:
        last = gen
        frame, gen = frame_and_next(gen)
        if frame is None or frame.f_lasti == -1:
            # Closed or just-created coroutine; assume this is
            # the bottom of the stack
            break
        frames.append((last, frame))
    return frames


def _match_trio_scopes_to_contexts(
    trio_scopes: Sequence[object], toplevel_contexts: Sequence[object]
) -> Dict[int, object]:
    """Given a sequence of Trio cancel scope and nursery objects,
    ordered from outermost to innermost, and a sequence of
    context manager objects, return a mapping indicating which
    Trio scope is managed (directly or indirectly) by which context
    manager.

    This is done using a breadth-first search of the GC references
    tree, currently limited to a depth of 5 links.  It lets us peek
    inside NurseryManager without directly depending on its existence,
    and also supports user-defined nursery wrappers such as
    "async with open_websocket():".
    """

    trio_scope_to_context = {}
    unmatched_trio_scopes = set(id(scope) for scope in trio_scopes)
    to_visit = collections.deque((ctx, ctx, 0) for ctx in toplevel_contexts)
    while unmatched_trio_scopes and to_visit:
        obj, root, depth = to_visit.popleft()
        if id(obj) in unmatched_trio_scopes:
            trio_scope_to_context[id(obj)] = root
            unmatched_trio_scopes.remove(id(obj))
        if depth < 5:
            to_visit.extend(
                (referent, root, depth + 1) for referent in gc.get_referents(obj)
            )
    return trio_scope_to_context


AssociatedObject = Union[
    trio.lowlevel.Task,
    trio_typing.Nursery,
    trio.CancelScope,
    Tuple[trio.lowlevel.Task, Awaitable[Any]],
    None,
]


def format_task_tree(
    task: Optional[trio.lowlevel.Task] = None, prefix: str = "", *, color: bool = False
) -> Iterator[Tuple[str, AssociatedObject]]:
    """Produce a textual depiction of the Trio task tree rooted at ``task``,
    or at the :func:`~trio.lowlevel.current_root_task` if ``task`` is None.

    The result is yielded one line at a time. Each line of
    human-readable text is provided alongside a reference to the task,
    nursery, cancel scope, or frame that it describes. Frames are
    yielded not as frame objects directly but rather as a (task,
    awaitable) tuple; you can easily get the frame from the awaitable
    and the awaitable can do other things too, like get
    weak-referenced.

    If you pass ``color=True``, the result strings will be brightened
    with ANSI color escape sequences for more evocative printing to a
    terminal.
    """

    # Map of module files to module names, for more compact tracebacking
    module_map = {
        mod.__file__: name
        for name, mod in sys.modules.items()
        if hasattr(mod, "__file__")
    }

    # Here lies the result of not wanting to add a dependency on a
    # colorization library:
    @attr.s(auto_attribs=True)
    class ColorOperator:
        code: str  # the color code, like "31" or "1;34"
        msg: str = ""  # the text being colorized

        # color | text => the text formatted with the color
        # (must be used in a format string to actually render)
        def __or__(self, incoming: str) -> "ColorOperator":
            return ColorOperator(code=self.code, msg=incoming)

        # color & color => the combination of the two colors
        def __and__(self, op: "ColorOperator") -> "ColorOperator":
            return ColorOperator(code=self.code + ";" + op.code)

        # Support f"{color:text}" meaning the (literal) "text" formatted
        # with the given (variable) color. If the part before the colon
        # already had some text incorporated, the part after the colon can
        # add to it or surround it:
        #   foo = "test"
        #   f"{color|foo:'s}" => colorized "test's"
        #   f"{color|foo:(_)}" => colorized "(test)"
        def _format(self, spec: str) -> str:
            if "_" in spec:
                return spec.replace("_", self.msg)
            else:
                return self.msg + spec

        if color:

            def __format__(self, spec: str) -> str:
                return f"\033[{self.code}m{self._format(spec)}\033[0m"

        else:

            def __format__(self, spec: str) -> str:
                return self._format(spec)

    yellow = ColorOperator("33")
    blue = ColorOperator("34")
    green = ColorOperator("32")
    bright = ColorOperator("1")

    # A shortcut for interpolating our ``prefix`` argument into
    # an f-string with color added, that keeps working even if
    # ``prefix`` gets changed:
    class ColoredPrefix:
        def __format__(self, spec: str) -> str:
            return (yellow | prefix).__format__(spec)

    cprefix = ColoredPrefix()

    def format_trio_scope(
        scope: object,
        function: str,
        filename: str,
        info: Optional[Tuple[Optional[str], int]],
    ) -> Iterator[Tuple[str, AssociatedObject]]:
        """Format a nursery or cancel scope, ``scope``."""
        nonlocal prefix
        varname: Optional[str]
        lineno: Union[int, str]
        if info is not None:
            varname, lineno = info
        else:
            varname, lineno, filename = None, "??", "??"
        nameinfo = f"{bright|varname} " if varname is not None else ""
        try:
            fileinfo = f"{module_map[filename]}:{lineno}"
        except KeyError:
            fileinfo = f"{filename}:{lineno}"
        where = f"in {bright|function} at {fileinfo}"

        def cancel_scope_info(
            cancel_scope: trio.CancelScope, color: ColorOperator
        ) -> str:
            bits = []
            if cancel_scope.cancel_called:
                bits.append("cancelled")
            elif cancel_scope.deadline != math.inf:
                bits.append(
                    "timeout in {:.2f}sec".format(
                        cancel_scope.deadline - trio.current_time()
                    )
                )
            if cancel_scope.shield:
                bits.append("shielded")
            return (": " if bits else "") + f"{color|', '.join(bits)}"

        if isinstance(scope, trio.CancelScope):
            yield (
                f"{cprefix}{blue|nameinfo:cancel scope _}{where}"
                + cancel_scope_info(scope, bright & blue),
                scope,
            )

        elif isinstance(scope, trio_typing.Nursery):
            yield (
                f"{cprefix}{green|nameinfo:nursery _}{where}"
                + cancel_scope_info(scope.cancel_scope, bright & green),
                scope,
            )
            for task in scope.child_tasks:
                yield from format_task_tree(task, prefix + "|-- ", color=color)
            yield f"{cprefix:_+--} nested child:", None
            prefix += "    "

        else:
            yield f"{cprefix}<!> unhandled {scope!r} {where}", None

    if task is None:
        task = trio.lowlevel.current_root_task()
        if task is None:
            return
    yield f"{cprefix}{yellow:task} {bright&yellow|task.name}:", task
    if prefix.endswith("-- "):
        prefix = prefix[:-3] + "   "

    frames = task_traceback(task)
    if not frames:
        yield f"{cprefix}<currently running, can't trace>", None
        yield f"{cprefix}", None
        return

    contexts_per_frame = [_contexts_active_in_frame(frame) for _, frame in frames]
    all_contexts = [ctx for sublist in contexts_per_frame for ctx, *info in sublist]
    all_trio_scopes = Deque[object]()

    if task.parent_nursery is not None:
        inherited_from_parent = len(task.parent_nursery._cancel_stack)  # type: ignore
    else:
        inherited_from_parent = 0
    cancel_scopes = list(
        reversed(task._cancel_stack[inherited_from_parent:])  # type: ignore
    )
    for nursery in task.child_nurseries:
        while cancel_scopes[-1] is not nursery.cancel_scope:
            all_trio_scopes.append(cancel_scopes.pop())
        # Don't include the nursery's cancel scope since we print it
        # as part of the nursery
        cancel_scopes.pop()
        all_trio_scopes.append(nursery)
    all_trio_scopes.extend(cancel_scopes[::-1])

    scope_map = _match_trio_scopes_to_contexts(all_trio_scopes, all_contexts)

    for (awaitable, frame), contexts_info in zip(frames, contexts_per_frame):
        filename, lineno, function, _, _ = inspect.getframeinfo(frame, context=0)
        if function == "_async_yield":
            # This is the innermost frame every time a task sleeps -- there's
            # no value to including it.
            continue

        context_to_info = {
            id(context): cast(Tuple[Optional[str], int], info)
            for context, *info in contexts_info
        }
        while all_trio_scopes:
            context = scope_map.get(id(all_trio_scopes[0]))
            if context is not None and id(context) not in context_to_info:
                # This context belongs to the next frame -- don't print it
                # in this frame
                break
            yield from format_trio_scope(
                all_trio_scopes.popleft(),
                function,
                filename,
                context_to_info.get(id(context)),
            )
        argvalues = inspect.getargvalues(frame)
        if argvalues.args and argvalues.args[0] == "self":
            function = f"{type(argvalues.locals['self']).__name__}.{function}"
        try:
            fileinfo = f"{module_map[filename]}:{lineno}"
        except KeyError:
            fileinfo = f"{filename}:{lineno}"
        yield f"{cprefix}{bright|function} at {fileinfo}", (task, awaitable)
    yield f"{cprefix if prefix.strip() else prefix}", None
