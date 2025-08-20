import sys
from types import FrameType
from dataclasses import dataclass, field
from typing import Any
import types

from collections.abc import Callable, Iterable
from circuits.utils.misc import OrderedSet


@dataclass(eq=False)
class CallNode[T]:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    count: int = 0
    parent: 'CallNode[T] | None' = None
    inputs: OrderedSet[T] = field(default_factory=OrderedSet[T])
    outputs: OrderedSet[T] = field(default_factory=OrderedSet[T])
    children: list['CallNode[T]'] = field(default_factory=list['CallNode[T]'])
    fn_counts: dict[str, int] = field(default_factory=dict[str, int])  # child fn name -> # direct calls in self
    skip: bool = False
    creation: T | None = None  # Instance created by this node, if any

    def create_child(self, fn_name: str) -> 'CallNode[T]':
        self.fn_counts[fn_name] = self.fn_counts.get(fn_name, 0) + 1
        child = CallNode(name=fn_name, count=self.fn_counts[fn_name]-1, parent=self)
        self.children.append(child)
        return child


def find_instances[T](obj: Any, target_type: type[T]) -> OrderedSet[T]:
    """Recursively find all T instances and their paths"""
    instances: OrderedSet[T] = OrderedSet()
    seen: set[Any] = set()

    def search(item: Any):

        # Handle circular references
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)

        # Add instances of target type
        if isinstance(item, target_type):
            instances.add(item)
            return  # assuming T does not contain T

        # Skip strings, bytes, and type annotations
        skippable = (str, bytes, type, types.GenericAlias, types.UnionType)
        if isinstance(item, skippable):
            return

        # Recurse on dicts and iterables
        if isinstance(item, dict):
            for _, value in item.items():  # type: ignore
                search(value)
        elif isinstance(item, Iterable):
            for elem in item:  # type: ignore
                search(elem)

    search(obj)

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
    tracked_type: type
    skip: set[str]
    collapse: set[str]

    def __post_init__(self) -> None:
        self.skip |= {'set_trace'}  # no need to track set_trace
        self.collapse |= {'<genexpr>'}  # avoids handling generator interactions with stack

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> CallNode[T]:
        def root_wrapper_fn(*args: Any, **kwargs: Any) -> Any:
            """Wraps a function call to avoid special handling of the root call"""
            return func(*args, **kwargs)
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
        root_wrapper = CallNode[T](name=func.__name__)
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
                    input_objects = [value for key, value in loc.items() if key != 'self']
                else:
                    input_objects = [value for _, value in loc.items()]
                node.inputs = find_instances(input_objects, self.tracked_type)

                # Tag tracked_type __init__ calls
                if is_creator:
                    # node.is_creator = True
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
