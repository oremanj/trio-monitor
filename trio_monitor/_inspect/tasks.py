@attr.s(auto_attribs=True, slots=True, eq=True, frozen=True)
class TaskTree:
    task: Optional[trio.lowlevel.Task]
    stack: Sequence[Frame]
    children: Sequence[TaskTree]
