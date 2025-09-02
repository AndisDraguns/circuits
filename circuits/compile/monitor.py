from sys import monitoring as mon
from types import CodeType, GenericAlias, UnionType
from typing import Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Iterable
import threading


type InstanceWithIndices[T] = tuple[T, list[int]]

@dataclass(eq=False)
class CallNode[T]:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    parent: 'CallNode[T] | None' = None
    children: list['CallNode[T]'] = field(default_factory=list['CallNode[T]'])
    inputs: list[InstanceWithIndices[T]] = field(default_factory=list[InstanceWithIndices[T]])
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


def find[T](obj: Any, target_type: type[T]) -> list[tuple[T, list[int]]]:
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
    collapse: set[str] = field(default_factory=set[str])
    stack: list[CallNode[T]] = field(default_factory = lambda: [CallNode[T]('root', parent = None)])
    _tracing_thread: int | None = None

    def __post_init__(self) -> None:
        """Avoids having to handle generator and context manager interactions with the stack"""
        self.collapse |= {'<genexpr>',  '__enter__', '__exit__'}

    def on_call(self, code: CodeType, offset: int):
        """Called when entering any function"""
        if threading.get_ident() != self._tracing_thread:
            return
        if '/site-packages/' in code.co_filename or '/lib/python' in code.co_filename:
            return
        if code.co_name in self.collapse:
            return
        if not self.stack:
            print(f"Error: stack is empty on {code.co_qualname} call")
            return
        node = self.stack[-1].create_child(code.co_name)
        self.stack.append(node)

    def on_return(self, code: CodeType, offset: int, retval: Any):
        """Called when exiting any function"""
        if threading.get_ident() != self._tracing_thread:
            return
        if '/site-packages/' in code.co_filename or '/lib/python' in code.co_filename:
            return
        if code.co_name in self.collapse:
            return
        if not self.stack:
            print(f"Error: stack is empty on {code.co_qualname} return")
            return
        node = self.stack.pop()
        node.outputs = find(retval, self.tracked_type)

    @property
    def root(self) -> CallNode[T]:
        assert len(self.stack) == 1  # only root should be left before/after tracing
        return self.stack[0]

    @contextmanager
    def trace(self):
        """Context manager to enable tracing"""
        # Set up tracing
        self._tracing_thread = threading.get_ident()
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
            self._tracing_thread = None
            mon.set_events(tool, 0)
            mon.free_tool_id(tool)



if __name__ == '__main__':
    # Example usage
    from circuits.neurons.core import Bit
    from circuits.utils.format import Bits
    from circuits.examples.keccak import Keccak
    def f(m: Bits, k: Keccak) -> list[Bit]:
        return k.digest(m).bitlist
    k = Keccak(c=10, l=0, n=2, pad_char='_')
    tracer = Tracer[Bit](Bit, collapse = {'__init__', 'outgoing', 'step'})
    msg = k.format("Reify semantics as referentless embeddings", clip=True)
    with tracer.trace():
        f(msg, k)
    print(tracer.root.tree())
