from types import FrameType
from circuits.neurons.core import Signal

import sys
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

Path = tuple[int|str, ...]
SignalPaths = list[tuple[Signal, Path]]

@dataclass(eq=False)
class CallNode:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    count: int = 0
    parent: 'CallNode | None' = None
    bot: int = 0  # Bottom height of the node in the call tree (relative to parent.top)
    top: int = 0  # Top height of the node in the call tree (relative to self.bot)
    left: int = 0  # left position of the node in the call tree (relative to parent.left)
    right: int = 0  # right position of the node in the call tree (relative to self.left)
    levels: list[int] = field(default_factory=list[int])  # level widths of the node in the call tree
    # creates_signal: bool = False  # True = this node or any of its subcalls create a Signal instance
    inputs: SignalPaths = field(default_factory=SignalPaths)
    outputs: SignalPaths = field(default_factory=SignalPaths)
    children: list['CallNode'] = field(default_factory=list['CallNode'])
    
    def info_str(self) -> str:
        """Returns a string representation of the node's info, excluding its children"""
        call_name = f"{self.name}-{self.count}"
        io = ""
        if self.inputs or self.outputs:
            io = f"({len(self.inputs)}→{len(self.outputs)})"
        bot_top = f"[b={self.bot}..t={self.top}]"
        left_right = f"[l={self.left}..r={self.right}]"
        res = f"{call_name} {io} {bot_top} {left_right}"
        return res

    def __str__(self, level: int = 0, hide: set[str] = set()) -> str:
        indent = "  " * level
        info = self.info_str()
        child_names = "".join(f"\n{c.__str__(level + 1, hide)}" for c in self.children if c.name not in hide)
        res = f"{indent}{info}{child_names}"
        return res

    @property
    def fpath(self) -> tuple['CallNode', ...]:
        """Returns the function path as a tuple of CallNodes from root to this node."""
        path: list['CallNode'] = []
        current: CallNode | None = self
        while current:
            path.append(current)
            current = current.parent
        return tuple(reversed(path))

    def to_dict(self) -> dict[str, Any]:
        """Recursively converts the node and its children to a dictionary."""
        return {
            "label": self.info_str(),    # Full label for the tooltip
            "bot": self.bot,
            "top": self.top,
            "left": self.left,
            "right": self.right,
            "children": [child.to_dict() for child in self.children if child.name not in {'gate'}],  # Exclude 'gate' nodes from children
        }

    def add(self, bot: int, top: int, width: int) -> int:
        """Adds a child node at bot-top height. width = child width.
        Updates self.levels widths. Returns the new child left position"""
        if len(self.levels) < top:
            self.levels.extend(0 for _ in range(len(self.levels), top))
        heights = list(range(bot, top))
        widths = [self.levels[h] for h in heights]
        new_left = max(widths) if widths else 0  # Find the maximum width at the heights
        self.levels[bot:top] = [new_left + width] * len(heights)  # Update all levels in the range to child right
        return new_left


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


def find_sibling_blocks(s_node: CallNode, s_parent_node: CallNode) -> tuple[CallNode, CallNode]:
    """
    Find sibling blocks within the last common ancestor. That is, ancestor of s_node and ancestor of s_parent_node,
    such that the two ancestors are different, but both are children of the last common ancestor.
    """
    s_fpath = s_node.fpath
    p_fpath = s_parent_node.fpath
    for i in range(min(len(s_fpath), len(p_fpath))):
        if s_fpath[i] != p_fpath[i]:
            # Found the first mismatch, return the sibling nodes
            return s_fpath[i], p_fpath[i]
    raise ValueError("Paths are identical, but the same gate call can not return both parent and child signal")

def process_gate_return(gate_node: CallNode, arg: Any) -> None:
    """Process gate return event"""
    # Idea for how to create blocks during ftrace:
    # Catch all events of gate return and:
    # - annotate the returned Signal s with current node node_fpath
    # - update all s parent fn blocks:
    #     - for each (node_fpath, parent_fpath) find their sibling blocks (n_sb, p_sb) within the last common ancestor
    #     - ensure that the current block is above the parent block
    # Now we have relative bot/top height for all blocks that generate Signals
    # TODO: handle blocks that do not generate Signals

    # Annotate the returned Signal with current node fpath
    n = gate_node
    s = arg  # signal returned by the gate
    s.trace.append(n)  # assumes that s.trace is [] except when turned into [gate_node] by this function

    if n.name != 'gate':
        raise ValueError(f"Expected gate node, got {n.name} instead")
    n.top += 1  # Set top height of the current node to 1, since gate is the only leaf node

    # Update n ancestors that are sibling blocks with s's parents
    s_parent_nodes = [p.trace[0] for p in s.source.incoming]  # nodes were assigned to parents when their nodes were created 
    for p in s_parent_nodes:
        n_sb, p_sb = find_sibling_blocks(n, p)
        if n_sb.bot < p_sb.top:  # current block must be above the parent block
            height_change = p_sb.top - n_sb.bot
            n_sb.bot += height_change
            n_sb.top += height_change

    n.right += 1  # Set the right position of the current node to 1, since gate is the only leaf node



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
            node = CallNode(name=func_name, count=counters[key], parent=parent)
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
            if func_name == 'gate' and len(stack) > 1:
                current = stack[-1]
                process_gate_return(current, arg)

            if func_name in collapse:
                collapse_depth -= 1
                return trace_handler

            # Record outputs
            if len(stack) > 1:
                current = stack.pop()
                current.outputs.extend(find_signals(arg, ()))

                # Sets node's coordinates. Only other update to it could have occurred before in process_gate_return.

                # Update current node's top height
                if current.children:
                    current_block_height = current.top - current.bot  # TODO: wait why - I guess top=/=height?
                    required_block_height = max([c.top for c in current.children])
                    if required_block_height > current_block_height:
                        current.top = current.bot + required_block_height
                # Update current node's left/right position
                if current.parent:
                    current_block_width = max(current.levels) if current.levels else current.right - current.left
                    horizontal_shift = current.parent.add(current.bot, current.top, current_block_width)
                    current.left += horizontal_shift
                    current.right = current.left + current_block_width

            # Root return
            else:
                root.outputs.extend(find_signals(arg, ()))
                # TODO: why not pop the root same as above?

                # Update current node's top and right
                if root.children:
                    root.top = max([c.top for c in root.children])
                    root.right = max(root.levels)

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


skip: set[str] = set()
collapse: set[str] = set()
collapse = {'__init__', '__post_init__', 'outgoing', 'step', 'reverse_bytes', 'lanes_to_state', 'format', 'bitlist', 'bitlist_to_msg',
            '<lambda>', '<genexpr>', 'msg_to_state', 'state_to_lanes', 'get_empty_lanes', 'get_round_constants', 'rho_pi',
            'copy_lanes', 'rot', 'xor', 'inhib', 'get_functions', '_bitlist_from_value', '_is_bit_list', 'from_str', 'const'}

from circuits.examples.keccak import Keccak
from circuits.neurons.core import Bit
def test() -> tuple[list[Bit], list[Bit]]:
    k = Keccak(c=20, l=1, n=1, pad_char='_')
    phrase = "Reify semantics as referentless embeddings"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)
    return message.bitlist, hashed.bitlist
io, tree = trace(test, skip=skip, collapse=collapse)
hide = {'gate'}
print(tree.__str__(hide=hide))

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