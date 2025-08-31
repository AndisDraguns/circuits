
"""Trace any function and collect statistics with sys.monitoring (Python 3.12+)"""

from sys import monitoring as mon
from types import CodeType, GenericAlias, UnionType
from typing import Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Iterable


type InstanceWithIndices[T] = tuple[T, list[int]]

@dataclass(eq=False)
class CallNode[T]:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    parent: 'CallNode[T] | None' = None
    children: list['CallNode[T]'] = field(default_factory=list['CallNode[T]'])
    outputs: list[InstanceWithIndices[T]] = field(default_factory=list[InstanceWithIndices[T]])
    count: int = 0
    counts: dict[str, int] = field(default_factory=dict[str, int])  # child call counts

    def create_child(self, name: str) -> 'CallNode[T]':
        self.counts[name] = self.counts.get(name, 0) + 1
        child = CallNode(name, parent=self, count=self.counts[name]-1)
        self.children.append(child)
        return child

    def __str__(self) -> str:
        return f"{self.name}-{self.count} →{len(self.outputs)}"

    def tree(self, level: int = 0, hide: set[str] = set()) -> str:
        child_strings = "".join(f"\n{c.tree(level+1, hide)}" for c in self.children if c.name not in hide)
        return f"{"  " * level}{str(self)}{child_strings}"


def find_instances[T](obj: Any, target_type: type[T]) -> list[tuple[T, list[int]]]:
    """Recursively find all T instances and their paths"""
    instances: list[tuple[T, list[int]]] = []
    seen: set[Any] = set()

    def search(item: Any, indices: list[int]):

        # Handle circular references
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)

        # Add instances of target type
        if isinstance(item, target_type):
            instances.append((item, indices))
            return  # assuming T does not contain T

        # Skip strings, bytes, and type annotations
        skippable = (str, bytes, type, GenericAlias, UnionType)
        if isinstance(item, skippable):
            return

        # Recurse on iterables
        elif isinstance(item, Iterable):
            if isinstance(item, dict):
                item = item.values()  # type: ignore
            for i, elem in enumerate(item):  # type: ignore
                next_indices = indices  # type: ignore
                if hasattr(item, '__len__') and len(item) > 1:  # type: ignore
                    next_indices += [i]
                # TODO: consider generators; kwargs indices to save memory
                search(elem, next_indices)

    search(obj, indices=[])

    return instances


@dataclass
class Tracer[T]:
    """Tracer with statistics collection"""
    tracked_type: type  # same as T
    stack: list[CallNode[T]] = field(default_factory = lambda: [CallNode[T]('root', parent = None)])
    collapse: set[str] = field(default_factory=set[str])

    def __post_init__(self) -> None:
        self.collapse |= {'<genexpr>',  '__enter__', '__exit__'}  # avoids handling generator interactions with stack
        self.collapse |= {'sandbagger', 'flat_sandbagger', 'xor_flat'}
        assert hasattr(self.tracked_type, '__init__')

    def on_call(self, code: CodeType, offset: int):
        """Called when entering any function"""
        if code.co_name in self.collapse:
            return
        node = self.stack[-1].create_child(code.co_name)
        self.stack.append(node)

    def on_return(self, code: CodeType, offset: int, retval: Any):
        """Called when exiting any function"""
        if code.co_name in self.collapse:
            return
        node = self.stack.pop()
        node.outputs = find_instances(retval, self.tracked_type)

    @contextmanager
    def trace(self):
        """Context manager to enable tracing"""
        tool = mon.DEBUGGER_ID
        pre = mon.events.PY_START
        post = mon.events.PY_RETURN
        mon.use_tool_id(tool, "tracer")
        mon.register_callback(tool, pre, self.on_call)
        mon.register_callback(tool, post, self.on_return)
        mon.set_events(tool, pre | post)
        try:
            yield
        finally:
            mon.set_events(tool, 0)
            mon.free_tool_id(tool)
        assert len(self.stack) == 1  # only root should be left



if __name__ == '__main__':
    # def fibonacci(n: int) -> int:
    #     if n <= 1:
    #         return n
    #     return fibonacci(n - 1) + fibonacci(n - 2)

    # tracer = Tracer[int](int)
    # with tracer.trace():
    #     fibonacci(3)
    # print(tracer.stack[0])

    from circuits.neurons.core import Bit
    from circuits.utils.format import Bits
    from circuits.examples.keccak import Keccak
    def f(m: Bits, k: Keccak) -> list[Bit]:
        return k.digest(m).bitlist
    k = Keccak(c=10, l=0, n=2, pad_char='_')
    tracer = Tracer[Bit](Bit, collapse = {'__init__', 'outgoing', 'step'})
    msg1 = k.format("Reify semantics as referentless embeddings", clip=True)
    with tracer.trace():
        f(msg1, k)
    print(tracer.stack[0].tree())



# import sys
# from dataclasses import dataclass, field
# from typing import Any
# from types import FrameType, GenericAlias, UnionType
# from collections.abc import Callable, Iterable


# if code == self.tracked_type.__init__.__code__:
#     node.is_creator = True


# @dataclass
# class FTracer[T]:
#     tracked_type: type  # same as T
#     collapse: set[str]  # exclude these functions
#     skip: set[str]  # exclude these functions and their subcalls

#     def __post_init__(self) -> None:
#         self.collapse |= {'<genexpr>', 'sandbagger', 'flat_sandbagger', 'xor_flat'}  # avoids handling generator interactions with stack
#         self.skip |= {'set_trace'}  # no need to track set_trace
#         if 'gate' in self.collapse:
#             self.collapse.remove('gate')
#             print("removed gate from collapse")

#     def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> CallNode[T]:
#         def root_wrapper_fn(*args: Any, **kwargs: Any) -> Any:
#             """Wraps a function call to avoid special handling of the root call"""
#             return func(*args, **kwargs)
#         trace = self.run_fn(root_wrapper_fn, *args, **kwargs)
#         return trace

#     def inits_tracked_type(self, frame: FrameType) -> bool:
#         """Returns True if the frame is an __init__ of the tracked type"""
#         loc = frame.f_locals
#         fn_name = frame.f_code.co_name
#         return fn_name == '__init__' and 'self' in loc and isinstance(loc['self'], self.tracked_type)

#     def run_fn(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> CallNode[T]:
#         """
#         Execute a function while building a tree of CallNodes tracking flow of T instances
#         Args: func: Function to trace; *args, **kwargs: Positional and keyword arguments to pass to func
#         """
#         root_wrapper = CallNode[T](func.__name__)
#         root_wrapper.inputs = find_instances(args + (kwargs,), self.tracked_type)
#         stack = [root_wrapper]
#         skipping = False
#         skipped_frame_id: int | None = None

#         def trace_handler(frame: FrameType, event: str, arg: Any):
#             nonlocal skipping, skipped_frame_id
            
#             if event == 'call':
#                 fn_name = frame.f_code.co_name

#                 # Handle unrecorded calls
#                 if fn_name == root_wrapper.name:
#                     return trace_handler
#                 if skipping:
#                     return trace_handler
#                 if fn_name in self.skip:
#                     skipping = True
#                     skipped_frame_id = id(frame)
#                     return trace_handler
#                 if fn_name in self.collapse:
#                     return trace_handler  # Continue tracing children, but don't create a node

#                 # Create a new node
#                 parent = stack[-1]
#                 node = parent.create_child(fn_name)
#                 stack.append(node)

#                 # Record inputs
#                 loc = frame.f_locals
#                 if fn_name == '__init__':  # Skip 'self' for __init__ methods as it's not fully initialized
#                     inputs = [value for key, value in loc.items() if key != 'self']
#                 else:
#                     inputs = [value for _, value in loc.items()]
#                 node.inputs = find_instances(inputs, self.tracked_type)

#                 return trace_handler


#             elif event == 'return':
#                 fn_name = frame.f_code.co_name

#                 # Handle unrecorded returns
#                 if fn_name == root_wrapper.name:
#                     return trace_handler
#                 if id(frame) == skipped_frame_id:
#                     skipping = False
#                     return trace_handler
#                 if skipping:
#                     return trace_handler
#                 if fn_name in self.collapse:
#                     return trace_handler

#                 # Remove node
#                 node = stack.pop()

#                 # Record outputs
#                 node.outputs = find_instances(arg, self.tracked_type)

#                 # Record T created by T.__init__
#                 if node.name == 'gate':
#                     assert len(node.outputs) == 1, f"gate {node.name} has {len(node.outputs)} outputs"
#                     node.creation = node.outputs[0][0]
#                     # TODO: make this more general

#                 return trace_handler

#             return trace_handler

#         # Execute with tracing
#         original_trace = sys.gettrace()
#         try:
#             set_trace(trace_handler)
#             result = func(*args, **kwargs)
#             root_wrapper.outputs = find_instances(result, self.tracked_type)
#         finally:
#             set_trace(original_trace)

#         return root_wrapper








# """Trace any function and collect statistics with sys.monitoring (Python 3.12+)"""

# from sys import monitoring as mon
# from types import CodeType
# from typing import Any
# from contextlib import contextmanager
# from dataclasses import dataclass, field


# @dataclass
# class TraceStats:
#     """Container for trace statistics"""
#     nr_calls: int = 0
#     nr_returns: int = 0

#     def summary(self):
#         """Print a summary of collected statistics"""
#         print(f"{self.nr_calls} calls, {self.nr_returns} returns")

# @dataclass
# class Tracer:
#     """Tracer with statistics collection"""
#     stats: TraceStats = field(default_factory=TraceStats)
        
#     def on_call(self, code: CodeType, offset: int):
#         """Called when entering any function"""
#         self.stats.nr_calls += 1
    
#     def on_return(self, code: CodeType, offset: int, retval: Any):
#         """Called when exiting any function"""
#         self.stats.nr_returns += 1
    
#     @contextmanager
#     def trace(self):
#         """Context manager to enable tracing"""
#         mon.use_tool_id(mon.DEBUGGER_ID, "tracer")
#         mon.register_callback(mon.DEBUGGER_ID, mon.events.PY_START, self.on_call)
#         mon.register_callback(mon.DEBUGGER_ID, mon.events.PY_RETURN, self.on_return)
#         mon.set_events(mon.DEBUGGER_ID, mon.events.PY_START | mon.events.PY_RETURN)
#         try:
#             yield
#         finally:
#             mon.set_events(mon.DEBUGGER_ID, 0)
#             mon.free_tool_id(mon.DEBUGGER_ID)


# if __name__ == '__main__':
#     def fibonacci(n: int) -> int:
#         if n <= 1:
#             return n
#         return fibonacci(n - 1) + fibonacci(n - 2)

#     tracer = Tracer()
#     with tracer.trace():
#         fibonacci(3)
#     tracer.stats.summary()
