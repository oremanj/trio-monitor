import attr
import collections
import ctypes
import dis
import gc
import inspect
import math
import sys
import traceback
import trio
import trio_typing
import warnings
from functools import partial
from types import FrameType, CodeType
from typing import (
    Any,
    Awaitable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
    cast,
)


@attr.s(auto_attribs=True, slots=True)
class ContextInfo:
    """Information about a sync or async context manager that's
    currently active in a frame."""

    #: True for an async context manager, False for a sync context manager.
    is_async: bool

    #: The context manager object itself (the ``foo`` in ``async with foo:``)
    manager: object = None

    #: The name to which the result of the context manager was assigned
    #: (``"bar"`` in ``async with foo as bar:``), or ``None`` if it wasn't
    #: assigned anywhere or if we couldn't determine where it was assigned.
    varname: Optional[str] = None

    #: The line number of the ``with`` or ``async with`` statement that
    #: entered the context manager, or ``None`` if we couldn't determine it.
    start_line: Optional[int] = None


def _contexts_active_by_referents(frame: FrameType) -> List[ContextInfo]:
    """Version of `contexts_active_in_frame` that relies only on
    `gc.get_referents`, and thus can be used on any Python interpreter
    that supports the `gc` module.

    This can't determine the `~ContextInfo.varname` or
    `~ContextInfo.start_line` members of `ContextInfo`, and it's
    possible to fool it in some unlikely circumstances (e.g., if you
    have a local variable that points directly to an ``__exit__`` or
    ``__aexit__`` method).
    """
    ret: List[ContextInfo] = []

    refs = gc.get_referents(frame)
    for idx, referent in enumerate(refs):
        if (
            isinstance(referent, types.MethodType)
            and referent.__func__.__name__ in ("__exit__", "__aexit__")
        ):
            # 'with' and 'async with' statements push a reference to the
            # __exit__ or __aexit__ method that they'll call when exiting.
            ret.append(
                ContextInfo(
                    is_async="a" in referent.__func__.__name__,
                    manager=referent.__self__,
                )
            )
    return ret


def _analyze_with_blocks(code: CodeType) -> Dict[int, ContextInfo]:
    """Analyze the bytecode of the given code object, returning a
    partially filled-in `ContextInfo` object for each ``with`` or
    ``async with`` block.

    Each key in the returned mapping is the bytecode offset of a
    ``WITH_CLEANUP_START`` instruction that ends a ``with`` or ``async
    with`` block. The corresponding value is a `ContextInfo` object
    appropriate to that block, with all fields except ``manager``
    filled in.
    """
    with_block_info: Dict[int, ContextInfo] = {}
    insns = list(dis.Bytecode(code))
    current_line = code.co_firstlineno
    for insn, inext in zip(insns[:-1], insns[1:]):
        if insn.starts_line is not None:
            current_line = insn.starts_line
        if insn.opname in ("SETUP_WITH", "SETUP_ASYNC_WITH"):
            if inext.opname == "STORE_FAST":
                store_to = inext.argval
            else:
                store_to = None
            cleanup_offset = insn.argval
            with_block_info[cleanup_offset] = ContextInfo(
                is_async=(insn.opname == "SETUP_ASYNC_WITH"),
                varname=store_to,
                start_line=current_line,
            )
    return with_block_info


@attr.s(auto_attribs=True)
class FrameDetails:
    @attr.s(auto_attribs=True)
    class FinallyBlock:
        handler: int  # bytecode offset where the finally handler starts
        level: int  # value stack depth at which handler begins execution

    blocks: List[FinallyBlock] = attr.Factory(list)
    stack: List[object] = attr.Factory(list)


def _inspect_frame_cpython(frame: FrameType) -> FrameDetails:
    details = FrameDetails()

    # This is the layout of the start of a frame object. It has a couple
    # fields we can't access from Python, especially f_valuestack and
    # f_stacktop.
    class FrameObjectStart(ctypes.Structure):
        _fields_ = [
            ("ob_refcnt", ctypes.c_ulong),  # reference count
            ("ob_type", ctypes.c_ulong),  # PyTypeObject*
            ("ob_size", ctypes.c_ulong),  # number of pointers after f_localsplus
            ("f_back", ctypes.c_ulong),  # PyFrameObject*
            ("f_code", ctypes.c_ulong),  # PyCodeObject*
            ("f_builtins", ctypes.c_ulong),  # PyDictObject*
            ("f_globals", ctypes.c_ulong),  # PyDictObject*
            ("f_locals", ctypes.c_ulong),  # PyObject*, some mapping
            ("f_valuestack", ctypes.c_ulong),  # PyObject**, points within self
            ("f_stacktop", ctypes.c_ulong),  # PyObject**, points within self
            # and then we start seeing differences between different
            # Python versions
        ]

    # Basic sanity checks cross-referencing the values we can get from Python
    # with their Python values
    frame_raw = FrameObjectStart.from_address(id(frame))
    refcnt = frame_raw.ob_refcnt
    assert refcnt + 1 == sys.getrefcount(frame)
    assert frame_raw.ob_type == id(type(frame))
    assert frame_raw.f_back == (id(frame.f_back) if frame.f_back is not None else 0)
    assert frame_raw.f_code == id(frame.f_code)
    assert frame_raw.f_globals == id(frame.f_globals)

    # The frame object has a fixed-length part followed by f_localsplus
    # which is a variable-length array of PyObject*. The array contains
    # co_nlocals + len(co_cellvars) + len(co_freevars) slots for
    # those things, followed by co_stacksize slots for the bytecode
    # interpreter stack. f_valuestack points at the beginning of the
    # stack part. Figure out where f_localsplus is. (It's a constant
    # offset from the start of the frame, but the constant differs
    # by Python version.)
    co = frame.f_code
    wordsize = ctypes.sizeof(ctypes.c_ulong)
    stack_start_offset = frame_raw.f_valuestack - id(frame)
    localsplus_offset = stack_start_offset - wordsize * (
        co.co_nlocals + len(co.co_cellvars) + len(co.co_freevars)
    )
    end_offset = stack_start_offset + wordsize * co.co_stacksize

    # Make sure our inferred size for the overall frame object matches
    # what Python says and what the ob_size field says. Note ob_size can
    # be larger than necessary due to frame object reuse.
    assert frame_raw.ob_size >= (end_offset - localsplus_offset) / wordsize
    assert end_offset == frame.__sizeof__()

    # Figure out what portion of the stack is actually valid, and extract
    # the PyObject pointers. We just store their addresses (id), not taking
    # references or anything.
    stack_top_offset = frame_raw.f_stacktop - id(frame)
    assert stack_start_offset <= stack_top_offset <= end_offset
    stack = [
        ctypes.c_ulong.from_address(id(frame) + offset).value
        for offset in range(stack_start_offset, stack_top_offset, wordsize)
    ]
    # Now stack[i] corresponds to f_stacktop[i] in C. Map addresses back
    # to actual objects, using gc.get_referents() so as not to crash if
    # we got the wrong addresses somehow.
    object_from_id_map = {id(obj): obj for obj in gc.get_referents(frame)}
    details.stack = [object_from_id_map.get(value) for value in stack]

    # Figure out the active context managers. Each context manager
    # pushes a block to a fixed-size block stack (20 12-byte entries,
    # this has been unchanged for ages) which is stored by value right
    # before f_localsplus. There's another frame field for the size of
    # the block stack.
    class PyTryBlock(ctypes.Structure):
        _fields_ = [
            # An opcode; the blocks we want are SETUP_FINALLY
            ("b_type", ctypes.c_int),
            # An offset in co.co_code; the blocks we want have a
            # WITH_CLEANUP_START opcode at this offset
            ("b_handler", ctypes.c_int),
            # An index on the value stack; if we're still in the body
            # of the with statement, the blocks we want have
            # an __exit__ or __aexit__ method at stack index b_level - 1
            ("b_level", ctypes.c_int),
        ]

    blockstack_offset = localsplus_offset - 20 * ctypes.sizeof(PyTryBlock)
    f_iblock = ctypes.c_int.from_address(id(frame) + blockstack_offset - 8)
    f_lasti = ctypes.c_int.from_address(id(frame) + blockstack_offset - 16)
    assert f_lasti.value == frame.f_lasti
    assert 0 <= f_iblock.value <= 20
    assert blockstack_offset > ctypes.sizeof(FrameObjectStart)

    blockstack_end_offset = blockstack_offset + (
        f_iblock.value * ctypes.sizeof(PyTryBlock)
    )
    assert blockstack_offset <= blockstack_end_offset <= localsplus_offset

    # Process blocks on the current block stack
    while blockstack_offset < blockstack_end_offset:
        block = PyTryBlock.from_address(id(frame) + blockstack_offset)
        assert (
            0 < block.b_type <= 257
            and (
                # EXCEPT_HANDLER blocks (type 257) have a bogus b_handler
                block.b_handler == -1
                if block.b_type == 257
                else 0 < block.b_handler < len(co.co_code)
            )
            and 0 <= block.b_level <= len(stack)
        )

        # Looks like a valid block -- is it one of our context managers?
        if (
            block.b_type == dis.opmap["SETUP_FINALLY"]
            and co.co_code[block.b_handler] == dis.opmap["WITH_CLEANUP_START"]
        ):
            # Yup. Still fully inside the block; use b_level to find the
            # __exit__ or __aexit__ method
            details.blocks.append(
                FrameDetails.FinallyBlock(handler=block.b_handler, level=block.b_level)
            )

        blockstack_offset += ctypes.sizeof(PyTryBlock)

    return details


_pypy_type_desc_from_index: List[str] = []
_pypy_type_index_from_id: Dict[int, int] = {}


def _fill_pypy_typemaps():
    assert sys.implementation.name == "pypy"
    import zlib

    for line in zlib.decompress(gc.get_typeids_z()).decode("ascii").splitlines():
        memberNNN, rest = line.split(None, 1)
        header, brace, fields = rest.partition(" { ")
        _pypy_type_desc_from_index.append(header)

    for idx, typeid in enumerate(gc.get_typeids_list()):
        _pypy_type_index_from_id[typeid] = idx


if sys.implementation.name == "pypy":
    _fill_pypy_typemaps()


def _pypy_typename(obj: object) -> str:
    return _pypy_type_desc_from_index[gc.get_rpy_type_index(obj)]


def _pypy_typename_from_first_word(first_word: int) -> str:
    if sys.maxsize > 2**32:
        mask = 0xffffffff
    else:
        mask = 0xffff
    return _pypy_type_desc_from_index[_pypy_type_index_from_id[first_word & mask]]


def _inspect_frame_pypy(frame: FrameType) -> FrameDetails:
    assert sys.implementation.name == "pypy"

    # Somewhere in the list of immediate referents of the frame is its
    # code object.
    frame_refs = gc.get_rpy_referents(frame)
    code_idx, = [idx for idx, ref in enumerate(frame_refs) if ref is frame.f_code]

    # The two referents immediately before the code object are
    # the last entry in the block list, followed by the value stack.
    # These are interp-level objects so we see them as opaque GcRefs.
    # We locate them by reference to the code object because the
    # earlier references might or might not be present (e.g., one depends
    # on whether the frame's f_locals have been accessed yet or not).
    assert code_idx >= 1
    valuestack_ref = frame_refs[code_idx - 1]
    assert isinstance(valuestack_ref, gc.GcRef)

    lastblock_ref: Optional[gc.GcRef] = None
    if code_idx >= 2:
        lastblock_ref = frame_refs[code_idx - 2]
        if "Block" not in _pypy_typename(lastblock_ref):
            # There are no blocks active in this frame. lastblock was
            # skipped when getting referents because it's null, so the
            # previous field (generator weakref or f_back) bled through.
            assert (
                _pypy_typename(lastblock_ref) == "GcStruct weakref"
                or "Frame" in _pypy_typename(lastblock_ref)
            )
            lastblock_ref = None
        else:
            assert isinstance(lastblock_ref, gc.GcRef)

    # The value stack's referents are everything on the value stack.
    # Unfortunately we can't rely on the indices here because 'del x'
    # leaves a null (not None) that will be skipped. We'll fill them
    # in from ctypes later. Note that this includes locals/cellvars/
    # freevars (at the start, in that order).
    valuestack = gc.get_rpy_referents(valuestack_ref)

    # The block list is a linked list in PyPy, unlike in CPython where
    # it's an array. The head of the list is the newest block.
    # Iterate through and unroll it into a list of GcRefs to blocks.
    blocks: List[gc.GcRef] = []
    if lastblock_ref is not None:
        blocks.append(lastblock_ref)
        while True:
            assert len(blocks) < 100
            more = gc.get_rpy_referents(blocks[-1])
            if not more:
                break
            blocks.extend(more)
        assert all("Block" in _pypy_typename(blk) for blk in blocks)
        # Reverse so the oldest block is at the beginning
        blocks = blocks[::-1]
        # Remove those that aren't FinallyBlocks -- those are the
        # only ones we care about (used for context managers too)
        blocks = [blk for blk in blocks if "FinallyBlock" in _pypy_typename(blk)]

    # This seems to be necessary to reliably make the object representations
    # correct before we start peeking at them.
    gc.collect()

    def unwrap_gcref(ref: gc.GcRef) -> "ctypes.pointer[ctypes.c_ulong]":
        ref_p = ctypes.pointer(ctypes.c_ulong.from_address(id(ref)))
        assert "W_GcRef" in _pypy_typename_from_first_word(ref_p[0])
        return ctypes.pointer(ctypes.c_ulong.from_address(ref_p[1]))

    # Fill in nulls in the value stack. This requires inspecting the
    # memory that backs the list object. An RPython list is two words
    # (typeid, length) followed by one word per element.
    def build_full_stack(refs: List[object]) -> Iterator[object]:
        stackdata_p = unwrap_gcref(valuestack_ref)
        assert _pypy_typename_from_first_word(stackdata_p[0]) == (
            "GcArray of * GcStruct object"
        )
        ref_iter = iter(refs)
        for idx in range(stackdata_p[1]):
            if stackdata_p[2 + idx] == 0:
                yield None
            else:
                try:
                    yield next(ref_iter)
                except StopIteration:
                    break

    details = FrameDetails(stack=list(build_full_stack(valuestack)))
    for block_ref in blocks:
        block_p = unwrap_gcref(block_ref)
        assert _pypy_typename_from_first_word(block_p[0]) == (
            "GcStruct pypy.interpreter.pyopcode.FinallyBlock"
        )
        details.blocks.append(
            FrameDetails.FinallyBlock(handler=block_p[1], level=block_p[3])
        )
    return details


def _contexts_active_by_trickery(frame: FrameType) -> List[ContextInfo]:
    """Version of `contexts_active_in_frame` that provides full information
    on tested versions of CPython and PyPy by accessing the block stack.
    This is an internal implementation detail so it may stop working as
    Python's internals change. The inspectors use lots of assertions so
    such failures will hopefully downgrade to the by_referents version,
    but there are no guarantees -- they might just segfault if we get
    really unlucky.
    """
    with_block_info = _analyze_with_blocks(frame.f_code)
    if sys.implementation.name == "cpython":
        details = _inspect_frame_cpython(frame)
    elif sys.implementation.name == "pypy":
        details = _inspect_frame_pypy(frame)
    else:
        raise NotImplementedError("trickery not supported on this interpreter")
    return [
        attr.evolve(
            with_block_info[blk.handler],
            manager=details.stack[blk.level - 1].__self__
        )
        for blk in details.blocks
    ]


"""
    # Decide whether it looks like we're in the middle of an
    # __aexit__, which would already have been popped from the block
    # stack.  In this case we can only get the coroutine, not the
    # original __aexit__ method.
    insns = list(dis.Bytecode(co.co_code))
    last_idx, = [
        idx for idx, insn in enumerate(insns) if frame.f_lasti == insn.offset
    ]
    if (
        last_idx >= 2
        and last_idx <= len(insns) - 2
        and [insn.opname for insn in insns[last_idx - 2 : last_idx + 2]]
        == ["WITH_CLEANUP_START", "GET_AWAITABLE", "LOAD_CONST", "YIELD_FROM"]
    ):
        coro_aexiting_now_id = stack[-1]
        coro_aexiting_now_offset = frame.f_lasti - 4

    # Now active_context_exit_ids contains (in theory) the addresses
    # of the __exit__/__aexit__ callables for each context manager
    # that's activated in this frame, from outermost to innermost. If
    # an async context manager is aexiting now, we additionally set
    # coro_aexiting_now_id to its id.  We'll add another layer of
    # security by getting the actual objects out of gc.get_referents()
    # rather than just casting the addresses.

    object_from_id_map = {
        id(obj): obj
        for obj in gc.get_referents(frame)
        if id(obj) in active_context_exit_ids or id(obj) == coro_aexiting_now_id
    }
    assert len(object_from_id_map) == len(active_context_exit_ids) + (
        coro_aexiting_now_id is not None
    )

    if coro_aexiting_now_id is not None:
        # Assume the context manager is the 1st coroutine argument.
        coro = object_from_id_map[coro_aexiting_now_id]
        args = inspect.getargvalues(coro.cr_frame)
        assert coro_aexiting_now_offset is not None
        active_context_info.append(
            attr.evolve(
                with_block_info[coro_aexiting_now_offset],
                manager=args.locals[args.args[0]],
            )
        )

    return active_context_info

"""


_can_use_trickery = sys.implementation.name in ("cpython", "pypy")
if _can_use_trickery:
    def _validate_trickery():
        from contextlib import contextmanager

        @contextmanager
        def noop():
            yield

        noop_cm = noop()

        def fn():
            with noop_cm as xyzzy:
                yield

        gen = fn()
        gen.send(None)
        try:
            contexts = _contexts_active_by_trickery(gen.gi_frame)
            assert len(contexts) == 1
            assert contexts[0].varname == "xyzzy" and contexts[0].manager is noop_cm
        except Exception as ex:
            warnings.warn(
                "Inspection trickery doesn't work on this interpreter: {!r}. "
                "Task tree printing will be less accurate. Please file a bug."
                .format(ex)
            )
            traceback.print_exc()
            return False
        return True

    _can_use_trickery = _validate_trickery()


def contexts_active_in_frame(frame: FrameType) -> List[ContextInfo]:
    """Inspects the frame object ``frame`` to try to determine which
    context managers are currently active; returns a list of
    `ContextInfo` objects describing the active context managers
    from outermost to innermost.

    The frame must be suspended at a yield or await. Raises AssertionError
    if we couldn't make sense of this frame.

    """
    if _can_use_trickery:
        try:
            return _contexts_active_by_trickery(frame)
        except Exception as ex:
            warnings.warn(
                "Inspection trickery failed on frame {!r}: {!r}. "
                "Task tree printing will be less accurate. Please file a bug."
                .format(frame, ex)
            )
            traceback.print_exc()
            return _contexts_active_by_referents(frame)
    else:
        return _contexts_active_by_referents(frame)


def frame_and_next(
    awaitable: Awaitable[Any]
) -> Tuple[Optional[FrameType], Optional[Awaitable[Any]]]:
    """Given an awaitable that is part of a suspended async call
    stack, return a tuple containing its active frame object and
    the other awaitable that it is currently awaiting.

    This supports coroutine objects, generator-based coroutines,
    async generator asend/athrow calls (both native and @async_generator),
    and coroutine wrappers. If the given awaitable isn't any of those,
    it returns (None, None).
    """

    typename = type(awaitable).__name__

    if typename == "coroutine":
        return awaitable.cr_frame, awaitable.cr_await  # type: ignore
    if typename == "generator":
        return awaitable.gi_frame, awaitable.gi_yieldfrom  # type: ignore
    if typename == "ANextIter":  # @async_generator awaitable
        return frame_and_next(awaitable._it)  # type: ignore
    if typename in ("async_generator_asend", "async_generator_athrow"):
        # native async generator awaitable, which holds a
        # reference to its agen but doesn't expose it
        for referent in gc.get_referents(awaitable):
            if hasattr(referent, "ag_frame"):
                return referent.ag_frame, referent.ag_await
    if typename == "coroutine_wrapper":
        # these refer to only one other object, the underlying coroutine
        for referent in gc.get_referents(awaitable):
            return frame_and_next(referent)
    return None, None


def task_traceback(task: trio.hazmat.Task) -> List[Tuple[Awaitable[Any], FrameType]]:
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
    trio.hazmat.Task,
    trio_typing.Nursery,
    trio.CancelScope,
    Tuple[trio.hazmat.Task, Awaitable[Any]],
    None,
]


def format_task_tree(
    task: Optional[trio.hazmat.Task] = None, prefix: str = "", *, color: bool = False
) -> Iterator[Tuple[str, AssociatedObject]]:
    """Produce a textual depiction of the Trio task tree rooted at ``task``,
    or at the :func:`~trio.hazmat.current_root_task` if ``task`` is None.

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
        task = trio.hazmat.current_root_task()
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
