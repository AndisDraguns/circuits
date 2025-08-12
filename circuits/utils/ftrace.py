from types import FrameType
import sys
from dataclasses import dataclass, field
from typing import Any

from collections.abc import Callable
from circuits.utils.format import Bits
from circuits.neurons.core import Signal


Path = tuple[int|str, ...]
SignalPaths = list[tuple[Signal, Path]]
ContainsBits = Any

@dataclass(eq=False)
class CallNode:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    count: int = 0
    parent: 'CallNode | None' = None
    is_root: bool = False
    is_live: bool = False  # live = generates signals that are downstream from inputs
    depth: int = 0  # Depth in the call tree
    bot: int = 0  # Bottom height of the node in the call tree (relative to parent.top)
    top: int = 0  # Top height of the node in the call tree (relative to self.bot)
    left: int = 0  # left position of the node in the call tree (relative to parent.left)
    right: int = 0  # right position of the node in the call tree (relative to self.left)
    levels: list[int] = field(default_factory=list[int])  # level widths of the node in the call tree
    inputs: SignalPaths = field(default_factory=SignalPaths)
    outputs: SignalPaths = field(default_factory=SignalPaths)
    children: list['CallNode'] = field(default_factory=list['CallNode'])
    x: int = -1  # Absolute x coordinate (leftmost edge)
    y: int = -1  # Absolute y coordinate (bottom edge)

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

    def get_relative_coordinates(self) -> tuple[float, float, float, float]:
        """Returns the relative coordinates (x, y, w, h) of the node.
        Each value is in percent relative to the parent dimension.
        x = leftmost edge; y = lowest edge"""
        if not self.parent:
            x, y, w, h = 0, 0, 100, 100
            return x, y, w, h
        parent_w = self.parent.right - self.parent.left
        parent_h = self.parent.top - self.parent.bot
        if parent_w == 0:
            x, w = 0, 100
        else:
            x = self.left / parent_w * 100
            w = (self.right - self.left) / parent_w * 100
        if parent_h == 0:
            y, h = 0, 100
        else:
            y = self.bot / parent_h * 100
            h = (self.top - self.bot) / parent_h * 100
        return x, y, w, h

    def set_absolute_coordinates(self) -> None:
        """Calculates and sets absolute coordinates"""
        if self.parent is None:  # is root
            self.x = 0
            self.y = 0
        else:
            self.x = self.left + self.parent.x
            self.y = self.bot + self.parent.y


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
        elif isinstance(item, Bits):
            try:
                for i, elem in enumerate(item.bitlist):
                    traverse(elem, current_path + (i,))
            except:
                return
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
    """
    Process gate return event
    # Idea for how to create blocks during ftrace:
    # Catch all events of gate return and:
    # - annotate the returned Signal s with current node node_fpath
    # - update all s parent fn blocks:
    #     - for each (node_fpath, parent_fpath) find their sibling blocks (n_sb, p_sb) within the last common ancestor
    #     - ensure that the current block is above the parent block
    # Now we have relative bot/top height for all blocks that generate Signals
    # TODO: handle blocks that do not generate Signals
    """

    # Annotate the returned Signal with current node fpath
    n = gate_node
    s = arg  # signal returned by the gate
    s.trace.append(n)  # assumes that s.trace is [] except when turned into [gate_node] by this function

    if n.name != 'gate':
        raise ValueError(f"Expected gate node, got {n.name} instead")
    n.top += 1  # Set top height of the current node to 1, since gate is the only leaf node

    # Update n ancestors that are sibling blocks with s's parents
    # nodes were assigned to parents when their nodes were created
    s_parent_nodes = [p.trace[0] for p in s.source.incoming if len(p.trace)>0]

    # One of the parent signals has no node -> it is an input. Therefore this node is live (downstream from inputs)
    if len(s_parent_nodes) != len(s.source.incoming) or any([p.is_live for p in s_parent_nodes]):
        n.is_live = True

    for p in s_parent_nodes:
        n_sb, p_sb = find_sibling_blocks(n, p)
        if n_sb.bot < p_sb.top:  # current block must be above the parent block
            height_change = p_sb.top - n_sb.bot
            n_sb.bot += height_change
            n_sb.top += height_change

    n.right += 1  # Set the right position of the current node to 1, since gate is the only leaf node


@dataclass
class TracerConfig:
    skip: set[str] = field(default_factory=set[str])
    collapse: set[str] = field(default_factory=set[str])


def set_trace(trace_func: Callable[[FrameType, str, Any], Any] | None) -> None:
    """Sets trace function using PyDev API. Avoids sys.settrace warnings."""
    if 'pydevd' in sys.modules:
        from pydevd_tracing import SetTrace  # type: ignore
        SetTrace(trace_func)
    else:
        sys.settrace(trace_func)


def tracer(func: Callable[..., Any], *args: Any,
          tracer_config: TracerConfig,
          **kwargs: Any
          ) -> tuple[Any, CallNode, int]:
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
    skip = tracer_config.skip | {'set_trace'}
    collapse = tracer_config.collapse
    
    # Initialize root with function inputs
    root = CallNode(name=func.__name__)
    inputs = find_signals(args, ('args',)) + find_signals(kwargs, ('kwargs',))
    root.inputs.extend(inputs)
    # for i, sp in enumerate(inputs):
    #     signal = sp[0]
    #     signal.trace.append(CallNode(name=f'inputs[{i}]', count=0, parent=None, depth=-1))
        # node = CallNode(name=func_name, count=counters[key], parent=parent, depth=parent.depth+1)
        # signal.name = f"arg_{i}"
        # signal.trace.append(root)

    # Tracking state
    stack = [root]
    counters: dict[tuple[int, str], int] = {}  # (parent_id, func_name) -> count
    skip_depth = 0
    collapse_depth = 0
    max_depth = 0

    def trace_handler(frame: FrameType, event: str, arg: Any):
        nonlocal skip_depth, collapse_depth, max_depth
        
        if event == 'call':
            func_name = frame.f_code.co_name

            # Handle skipping and collapse
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
            node = CallNode(name=func_name, count=counters[key], parent=parent, depth=parent.depth+1)
            parent.children.append(node)
            stack.append(node)

            # Extract function arguments and find Signals
            for name, value in frame.f_locals.items():
                node.inputs.extend(find_signals(value, (name,)))
                
        elif event == 'return':
            func_name = frame.f_code.co_name
            
            if func_name == 'gate' and len(stack) > 1:
                node = stack[-1]
                process_gate_return(node, arg)

            # Handle skipping and collapse
            if skip_depth > 0:
                skip_depth -= 1
                return trace_handler
            if func_name in collapse:
                collapse_depth -= 1
                return trace_handler

            # Record outputs
            if len(stack) > 1:
                node = stack.pop()
                node.outputs.extend(find_signals(arg, ()))

                # Sets node's coordinates. Only other update to it could have occurred before in process_gate_return.
                set_top(node)
                set_left_right(node)

                if node.depth > max_depth:
                    max_depth = node.depth

                if any([c.is_live for c in node.children]):
                    node.is_live = True

            # Root return
            else:
                root.outputs.extend(find_signals(arg, ()))
                # TODO: why not pop the root same as above?

                # Update current node's top and right
                if root.children:
                    root.top = max([c.top for c in root.children])
                    try:
                        root.right = max(root.levels)
                    except:
                        print(root.children)
                        print(root.levels)

        return trace_handler

    # Execute with tracing
    original_trace = sys.gettrace()
    try:
        set_trace(trace_handler)
        result = func(*args, **kwargs)
    finally:
        set_trace(original_trace)
        
    return (result), root, max_depth


def set_top(node: CallNode) -> None:
    """Sets the top height of the node based on its children"""
    if not node.children:
        return
    block_height = max([c.top for c in node.children])
    if node.name == 'gate':
        block_height = 1
    node.top = node.bot + block_height


def set_left_right(node: CallNode) -> None:
    """Sets the left and right position of the node based on its parent"""
    if not node.parent:
        return
    current_block_width = max(node.levels) if node.levels else node.right - node.left
    horizontal_shift = node.parent.add(node.bot, node.top, current_block_width)
    node.left += horizontal_shift
    node.right = node.left + current_block_width



if __name__ == '__main__':
    skip: set[str] = set()
    collapse = {'__init__', '__post_init__', 'outgoing', 'step', 'reverse_bytes', 'lanes_to_state', 'format', 'bitlist', 'bitlist_to_msg',
                '<lambda>', '<genexpr>', 'msg_to_state', 'state_to_lanes', 'get_empty_lanes', 'get_round_constants', 'rho_pi',
                'copy_lanes', 'rot', 'xor', 'inhib', 'get_functions', '_bitlist_from_value', '_is_bit_list', 'from_str', 'const'}
    tracer_config = TracerConfig(skip=skip, collapse=collapse)

    # from circuits.examples.keccak import Keccak
    # from circuits.neurons.core import Bit
    # from circuits.utils.format import Bits
    # def test() -> list[Bit]:
    #     phrase = "Reify semantics as referentless embeddings"
    #     k = Keccak(c=20, l=1, n=1, pad_char='_')
    #     message = k.format(phrase, clip=True)
    #     hashed = k.digest(message)
    #     return hashed.bitlist
    # _, root, _ = tracer(test, tracer_config=tracer_config)

    from circuits.examples.keccak import Keccak
    from circuits.neurons.core import Bit
    from circuits.utils.format import Bits
    def test(message: Bits, k: Keccak) -> list[Bit]:
        hashed = k.digest(message)
        return hashed.bitlist
    k = Keccak(c=20, l=1, n=1, pad_char='_')
    phrase = "Reify semantics as referentless embeddings"
    message = k.format(phrase, clip=True)
    _, root, _ = tracer(test, tracer_config=tracer_config, message=message, k=k)
    
    
    hide = {'gate'}
    # hide: set[str] = set()
    print(root.__str__(hide=hide))

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



