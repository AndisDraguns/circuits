from types import FrameType
from circuits.neurons.core import Signal

import sys
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

Path = tuple[int|str, ...]
SignalPaths = list[tuple[Signal, Path]]

@dataclass
class CallNode:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    count: int = 0
    inputs: SignalPaths = field(default_factory=SignalPaths)
    outputs: SignalPaths = field(default_factory=SignalPaths)
    children: list['CallNode'] = field(default_factory=list['CallNode'])
    
    def __str__(self, level: int = 0):
        indent = "  " * level
        s = f"{indent}{self.name}-{self.count}"
        if self.inputs or self.outputs:
            s += f" ({len(self.inputs)}→{len(self.outputs)})"
        return s + "".join(f"\n{child.__str__(level + 1)}" for child in self.children)

    def to_dict(self) -> dict[str, str|list[Any]]:
        """Recursively converts the node and its children to a dictionary."""
        # Create a display-friendly label for the node
        label = f"{self.name}-{self.count}"
        if self.inputs or self.outputs:
            label += f" ({len(self.inputs)}→{len(self.outputs)})"
        return {
            "name": label,
            "children": [child.to_dict() for child in self.children]
        }

def find_signals(obj: Any, path: Path = ()) -> SignalPaths:
    """Recursively find all Signal instances and their paths"""
    signals: SignalPaths = []
    seen: set[Any] = set()  # Handle circular references
    def traverse(item: Any, current_path: Path):
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)
        if isinstance(item, Signal):
            signals.append((item, current_path))
        elif isinstance(item, (list, tuple)):
            for i, elem in enumerate(item):  # type: ignore
                traverse(elem, current_path + (i,))
        elif isinstance(item, dict):
            for key, value in item.items():  # type: ignore
                traverse(value, current_path + (key,))  # type: ignore
    traverse(obj, path)
    return signals


def set_trace(trace_func: Callable[[FrameType, str, Any], Any] | None) -> None:
    """Sets trace function using PyDev API. Avoids sys.settrace warnings."""
    if 'pydevd' in sys.modules:
        from pydevd_tracing import SetTrace  # type: ignore
        SetTrace(trace_func)
    else:
        sys.settrace(trace_func)


def trace(func: Callable[..., Any], *args: Any,
          skip: set[str] = set(),
          collapse: set[str] = set(),
          **kwargs: Any
          ) -> tuple[Any, CallNode]:
    """
    Execute function while building a tree of CallNodes tracking Signal flow.
    
    Args:
        func: Function to trace
        *args: Positional arguments to pass to func
        skip: Set of function names to skip
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        tuple: (result, root_node)
    """
    skip = skip | {'set_trace'}
    
    # Initialize root with function inputs
    root = CallNode(name=func.__name__)
    root.inputs.extend(find_signals(args, ('args',)))
    root.inputs.extend(find_signals(kwargs, ('kwargs',)))
    
    # Tracking state
    stack = [root]
    counters: dict[tuple[int, str], int] = {}  # (parent_id, func_name) -> count
    skip_depth = 0
    collapse_depth = 0

    def trace_handler(frame: FrameType, event: str, arg: Any):
        nonlocal skip_depth, collapse_depth
        
        if event == 'call':
            func_name = frame.f_code.co_name
            
            # Handle skipping
            if skip_depth > 0 or func_name in skip:
                skip_depth += 1
                return trace_handler

            if func_name in collapse:
                collapse_depth += 1
                return trace_handler # Continue tracing children, but don't create a node
                
            # Skip the root function call itself
            if len(stack) == 1 and func_name == root.name:
                return trace_handler
            
            # Create child node
            parent = stack[-1]
            key = (id(parent), func_name)
            counters[key] = counters.get(key, -1) + 1
            node = CallNode(name=func_name, count=counters[key])
            parent.children.append(node)
            stack.append(node)

            # Extract function arguments and find Signals
            for name, value in frame.f_locals.items():
                node.inputs.extend(find_signals(value, (name,)))
                
        elif event == 'return':
            if skip_depth > 0:
                skip_depth -= 1
                return trace_handler

            func_name = frame.f_code.co_name
            if func_name in collapse:
                collapse_depth -= 1
                return trace_handler

            # Record outputs
            if len(stack) > 1:
                current = stack.pop()
                current.outputs.extend(find_signals(arg, ()))

            # Root return
            else:
                root.outputs.extend(find_signals(arg, ()))

        return trace_handler

    # Execute with tracing
    original_trace = sys.gettrace()
    try:
        set_trace(trace_handler)
        result = func(*args, **kwargs)
    finally:
        set_trace(original_trace)
        
    return (result), root



# ---- DEMO ----

skip={'gate'}
# skip = set()
# collapse = {'__init__', '__post_init__', 'gate', 'reverse_bytes', 'lanes_to_state', 'format', 'bitlist', 'bitlist_to_msg',
#             '<lambda>', '<genexpr>', 'msg_to_state', 'state_to_lanes', 'get_empty_lanes', 'const', 'get_round_constants', 'rho_pi',
#             'copy_lanes', 'rot', 'xor', 'not_', 'inhib', 'get_functions', '_bitlist_from_value', '_is_bit_list', 'from_str'}
collapse = {'__init__', '__post_init__', 'gate', 'reverse_bytes', 'lanes_to_state', 'format', 'bitlist', 'bitlist_to_msg',
            '<lambda>', '<genexpr>', 'msg_to_state', 'state_to_lanes', 'get_empty_lanes', 'const', 'get_round_constants', 'rho_pi',
            'copy_lanes', 'rot', 'xor', 'not_', 'inhib', 'get_functions', '_bitlist_from_value', '_is_bit_list', 'from_str'}


from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit
def test() -> tuple[list[Bit], list[Bit]]:
    k = Keccak(c=20, l=3, n=12, pad_char='_')
    phrase = "Reify semantics as referentless embeddings"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)
    return message.bitlist, hashed.bitlist
io, tree = trace(test, skip=skip, collapse=collapse)
# graph = compiled_from_io(inp, out)
print(tree)

# from circuits.neurons.core import const
# from circuits.neurons.operations import xors, ands
# from circuits.sparse.compile import compiled_from_io
# def test():
#     a = const('110')
#     b = const('101')
#     c = const('111')
#     res1 = xors([a, b])
#     res2 = xors([b, c]) 
#     res3 = ands([res1, res2])
#     return a+b+c, res3
# inp, out = test()
# io, tree = trace(test, skip=skip)
# inp, out = io
# graph = compiled_from_io(inp, out)
# print(tree)