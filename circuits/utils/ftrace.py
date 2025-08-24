import sys
from dataclasses import dataclass, field
from typing import Any
from types import FrameType, GenericAlias, UnionType
from collections.abc import Callable, Iterable

type InstanceWithIndices[T] = tuple[T, list[int]]


@dataclass(eq=False)
class CallNode[T]:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    parent: 'CallNode[T] | None' = None
    children: list['CallNode[T]'] = field(default_factory=list['CallNode[T]'])
    inputs: list[InstanceWithIndices[T]] = field(default_factory=list[InstanceWithIndices[T]])
    outputs: list[InstanceWithIndices[T]] = field(default_factory=list[InstanceWithIndices[T]])
    creation: T | None = None  # T instance constructed iff this node is T.__init__
    count: int = 0
    counts: dict[str, int] = field(default_factory=dict[str, int])  # child call counts

    def create_child(self, fn_name: str) -> 'CallNode[T]':
        self.counts[fn_name] = self.counts.get(fn_name, 0) + 1
        child = CallNode(fn_name, parent=self, count=self.counts[fn_name]-1)
        self.children.append(child)
        return child


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
                next_indices = indices + [i] if len(item) > 1 else indices
                search(elem, next_indices)
                # TODO: kwargs indices to save memory

    search(obj, indices=[])

    return instances


def set_trace(trace_func: Callable[[FrameType, str, Any], Any] | None) -> None:
    """Sets trace function using PyDev API. Avoids sys.settrace warnings."""
    if 'pydevd' in sys.modules:
        from pydevd_tracing import SetTrace  # type: ignore
        SetTrace(trace_func)
    else:
        sys.settrace(trace_func)


@dataclass
class FTracer[T]:
    tracked_type: type  # same as T
    collapse: set[str]  # exclude these functions
    skip: set[str]  # exclude these functions and their subcalls

    def __post_init__(self) -> None:
        self.collapse |= {'<genexpr>'}  # avoids handling generator interactions with stack
        self.skip |= {'set_trace'}  # no need to track set_trace

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> CallNode[T]:
        def root_wrapper_fn(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            def root_wrapper_fn_2(*args: Any, **kwargs: Any) -> Any:
                """Wraps a function call to avoid special handling of the root call"""
                return func(*args, **kwargs)
            return root_wrapper_fn_2(*args, **kwargs)
        trace = self.run_fn(root_wrapper_fn, *args, **kwargs)
        return trace

    def inits_tracked_type(self, frame: FrameType) -> bool:
        """Returns True if the frame is an __init__ of the tracked type"""
        loc = frame.f_locals
        fn_name = frame.f_code.co_name
        return fn_name == '__init__' and 'self' in loc and isinstance(loc['self'], self.tracked_type)

    def run_fn(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> CallNode[T]:
        """
        Execute a function while building a tree of CallNodes tracking flow of T instances
        Args: func: Function to trace; *args, **kwargs: Positional and keyword arguments to pass to func
        """
        root_wrapper = CallNode[T](func.__name__)
        root_wrapper.inputs = find_instances(args + (kwargs,), self.tracked_type)
        stack = [root_wrapper]
        skipping = False
        skipped_frame_id: int | None = None

        def trace_handler(frame: FrameType, event: str, arg: Any):
            nonlocal skipping, skipped_frame_id
            
            if event == 'call':
                fn_name = frame.f_code.co_name

                # Handle unrecorded calls
                if fn_name == root_wrapper.name:
                    return trace_handler
                if skipping:
                    return trace_handler
                if fn_name in self.skip:
                    skipping = True
                    skipped_frame_id = id(frame)
                    return trace_handler
                is_creator = self.inits_tracked_type(frame)
                if fn_name in self.collapse and not is_creator:
                    return trace_handler  # Continue tracing children, but don't create a node

                # Create a new node
                parent = stack[-1]
                node = parent.create_child(fn_name)
                stack.append(node)

                # Record inputs
                loc = frame.f_locals
                if fn_name == '__init__':  # Skip 'self' for __init__ methods as it's not fully initialized
                    inputs = [value for key, value in loc.items() if key != 'self']
                else:
                    inputs = [value for _, value in loc.items()]
                node.inputs = find_instances(inputs, self.tracked_type)

                # Record T created by T.__init__
                if is_creator:
                    node.creation = loc['self']

                return trace_handler


            elif event == 'return':
                fn_name = frame.f_code.co_name

                # Handle unrecorded returns
                if fn_name == root_wrapper.name:
                    return trace_handler
                if id(frame) == skipped_frame_id:
                    skipping = False
                    return trace_handler
                if skipping:
                    return trace_handler
                if fn_name in self.collapse and not self.inits_tracked_type(frame):
                    return trace_handler

                # Remove node
                node = stack.pop()

                # Record outputs
                node.outputs = find_instances(arg, self.tracked_type)
                return trace_handler

            return trace_handler

        # Execute with tracing
        original_trace = sys.gettrace()
        try:
            set_trace(trace_handler)
            _ = func(*args, **kwargs)
        finally:
            set_trace(original_trace)

        return root_wrapper
