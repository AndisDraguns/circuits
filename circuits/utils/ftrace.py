import sys
from types import FrameType
from dataclasses import dataclass, field
from typing import Any

from collections.abc import Callable, Generator
from circuits.utils.format import Bits
from circuits.neurons.core import Signal
from circuits.utils.misc import OrderedSet


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
    inputs: OrderedSet[Signal] = field(default_factory=OrderedSet[Signal])
    outputs: OrderedSet[Signal] = field(default_factory=OrderedSet[Signal])
    children: list['CallNode'] = field(default_factory=list['CallNode'])

    # post-processing info:
    x: int = -1  # Absolute x coordinate (leftmost edge)
    y: int = -1  # Absolute y coordinate (bottom edge)
    full_name: str | None = None  # Full name of the node in the call tree
    out_str: str = ""  # String representation of the outputs
    highlight: bool = False  # Whether to highlight this node
    outdiff: str = ""

    # connections info
    input_sources: list[tuple['CallNode|None', int]] = field(default_factory=list[tuple['CallNode|None', int]])  # (source node, its output index) for each input

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

    def full_info(self) -> str:
        s = f"name-count: {self.name}-{self.count}\n"
        s += f"io: ({len(self.inputs)}→{len(self.outputs)})\n"
        s += f"full_name: {self.full_name}\n"
        s += f"depth of nesting: {self.depth}\n"
        s += f"x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}\n"
        s += f"is_live: {self.is_live}, highlight: {self.highlight}\n"
        s += f"out_str: '{self.out_str}'\n"
        if self.outdiff:
            s += f"outdiff: '{self.outdiff}'\n"
        return s

    @property
    def fpath(self) -> tuple['CallNode', ...]:
        """Returns the function path as a tuple of CallNodes from root to this node."""
        path: list['CallNode'] = []
        current: CallNode | None = self
        while current:
            path.append(current)
            current = current.parent
        return tuple(reversed(path))

    @property
    def h(self) -> int:
        """Height in absolute units"""
        return self.top - self.bot
    
    @property
    def w(self) -> int:
        """Width in absolute units"""
        assert self.right - self.left >= 0, f"self.right - self.left = {self.right - self.left}, {self.name}"
        return self.right - self.left

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

    def set_absolute_coordinates(self) -> None:
        """Calculates and sets absolute coordinates"""
        if self.parent is None:  # is root
            self.x = 0
            self.y = 0
        else:
            self.x = self.left + self.parent.x
            self.y = self.bot + self.parent.y


def find_signals(obj: Any) -> OrderedSet[Signal]:
    """Recursively find all Signal instances and their paths"""
    signals: OrderedSet[Signal] = OrderedSet()
    seen: set[Any] = set()  # Handle circular references
    def traverse(item: Any):
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)
        if isinstance(item, Signal):
            signals.add(item)
        elif isinstance(item, (list, tuple)):
            for elem in item:  # type: ignore
                traverse(elem)
        elif isinstance(item, Bits):
            try:
                for elem in item.bitlist:
                    traverse(elem)
            except:
                return
        elif isinstance(item, dict):
            for key, value in item.items():  # type: ignore
                traverse(value)  # type: ignore
    traverse(obj)
    return signals


def get_lca_children_split(x: CallNode, y: CallNode) -> tuple[CallNode, CallNode]:
    """
    Find the last common ancestor of x and y.
    Then returns its two children a and b that are on paths to x and y respectively.
    """
    x_path = x.fpath
    y_path = y.fpath
    for i in range(min(len(x_path), len(y_path))):
        if x_path[i] != y_path[i]:
            return x_path[i], y_path[i]  # Found the first mismatch, return lca_child_to_x, lca_child_to_y
    raise ValueError("x and y are on the same path to root")


def update_ancestor_heights(n: CallNode) -> None:
    """
    On return of a node n, set node's height to be after its inputs, update ancestor heights if necessary.
    For each input, its creator gate node g is located. For 
    """
    # ignore gate subcalls, since is_live calculation assumes that gate is a leaf node
    if 'gate' in [p.name for p in n.fpath[:-1]]:
        return

    for inp in n.inputs:
        if len(inp.trace) == 0:
            n.is_live = True  # no trace -> input created outside of fn -> node is live (downstream from inputs)
            continue
        g = inp.trace[0]
        n_ancestor, g_ancestor = get_lca_children_split(n, g)
        if g_ancestor.is_live:
            n.is_live = True
        if n_ancestor.bot < g_ancestor.top:  # current block must be above the parent block
            height_change = g_ancestor.top - n_ancestor.bot
            n_ancestor.bot += height_change
            n_ancestor.top += height_change

    # TODO: can we update max shifting parent instead of fully looping over all inputs?
    # TODO: add copies if input gates are distant


def process_gate_return(g: CallNode) -> None:
    if len(g.outputs) == 0:
        return
    s = list(g.outputs)[0]  # Get the output signal of the gate
    assert len(g.outputs) == 1 and g.name == 'gate'
    assert s.trace == [], f"s.trace should be [] before creation of gate node, got {s.trace}"
    s.trace.append(g)
    g.top += 1  # Set top height of g to 1, since gate is the only leaf node
    g.right += 1  # Set the right position of g to 1, since gate is the only leaf node


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
    if len(node.outputs) > current_block_width:
        current_block_width = len(node.outputs)
    horizontal_shift = node.parent.add(node.bot, node.top, current_block_width)
    node.left += horizontal_shift
    node.right = node.left + current_block_width



@dataclass
class Trace:
    root: CallNode
    max_depth: int
    input_args: Any
    input_kwargs: Any
    output: Any = None

    def highlight_differences(self, other: 'Trace') -> None:
        """
        Highlights the differences between two call trees.
        Sets 'highlight' flag in for each call node that differs from the corresponding node in the other tree.
        """
        gen1 = walk_generator(other.root)
        gen2 = walk_generator(self.root)
        for node1, node2 in zip(gen1, gen2):
            assert node1.full_name == node2.full_name, f"Node names do not match: {node1.full_name} != {node2.full_name}"
            if node1.out_str != node2.out_str:
                node2.highlight = True
                node2.outdiff = "".join([' ' if s1==s2 else s1 for s1, s2 in zip(node1.out_str, node2.out_str)])


def set_trace(trace_func: Callable[[FrameType, str, Any], Any] | None) -> None:
    """Sets trace function using PyDev API. Avoids sys.settrace warnings."""
    if 'pydevd' in sys.modules:
        from pydevd_tracing import SetTrace  # type: ignore
        SetTrace(trace_func)
    else:
        sys.settrace(trace_func)


@dataclass
class Tracer:
    skip: set[str] = field(default_factory=set[str])
    collapse: set[str] = field(default_factory=set[str])

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Trace:
        trace = self.run_fn(func, *args, **kwargs)
        trace = self.post_process_trace(trace)
        return trace

    def run_fn(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Trace:
 
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
        skip = self.skip | {'set_trace'}
        collapse = self.collapse
        
        # Initialize root with function inputs
        root = CallNode(name=func.__name__)
        root.inputs = find_signals(args) + find_signals(kwargs)

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
                inputs = [value for _, value in frame.f_locals.items()]
                node.inputs = find_signals(inputs)
                
                    
            elif event == 'return':
                func_name = frame.f_code.co_name
                
                if func_name == 'gate' and len(stack) > 1:
                    node = stack[-1]
                    node.outputs = find_signals(arg)
                    process_gate_return(node)

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
                    node.outputs = find_signals(arg)
                    # node.output_set = {s for s, _ in node.outputs}

                    # Sets node's coordinates. Only other update to it could have occurred before in process_gate_return.
                    update_ancestor_heights(node)
                    set_top(node)
                    # set_connections(node)
                    set_left_right(node)

                    if node.depth > max_depth:
                        max_depth = node.depth

                    if any([c.is_live for c in node.children]):
                        node.is_live = True

                # Root return
                else:
                    root.outputs = find_signals(arg)
                    # root.outputs.extend(find_signals(arg, ()))
                    # root.output_set = {s for s, _ in root.outputs}
                    # TODO: why not pop the root same as above?

                    # set_connections(root)
                    # Update current node's top and right
                    if root.children:
                        root.top = max([c.top for c in root.children])
                        try:
                            root.right = max(root.levels)
                        except:
                            print(root.children)
                            print(root.levels)

                    if any([c.is_live for c in root.children]):
                        root.is_live = True

            return trace_handler

        # Execute with tracing
        original_trace = sys.gettrace()
        try:
            set_trace(trace_handler)
            result = func(*args, **kwargs)
        finally:
            set_trace(original_trace)
            
        return Trace(root, max_depth, args, kwargs, result)


    def post_process_trace(self, trace: Trace) -> Trace:
        """Processes the call tree"""
        for n in walk_generator(trace.root):
            if n.parent is None:  # is root
                n.full_name = f"{n.name}-{n.count}"
            else:
                n.full_name = f"{n.parent.full_name}.{n.name}-{n.count}"
            n.out_str = Bits(list(n.outputs)).bitstr
            n.set_absolute_coordinates()
        return trace


def walk_generator(node: CallNode) -> Generator[CallNode, None, None]:
    """Walks the call tree and yields each node."""
    if not node:
        return
    yield node
    for child in node.children:
        yield from walk_generator(child)


def get_output_levels(root: CallNode) -> list[OrderedSet[Signal]]:
    """Gets the output levels of the call tree."""
    levels: list[OrderedSet[Signal]] = [OrderedSet() for _ in range(root.top+1)]
    for n in walk_generator(root):
        for out in n.outputs:
            levels[n.top].add(out)
    return levels


if __name__ == '__main__':
    skip: set[str] = set()
    collapse = {'__init__', '__post_init__', 'outgoing', 'step', 'reverse_bytes', 'lanes_to_state', 'format', 'bitlist', 'bitlist_to_msg',
                '<lambda>', '<genexpr>', 'msg_to_state', 'state_to_lanes', 'get_empty_lanes', 'get_round_constants', 'rho_pi',
                'copy_lanes', 'rot', 'xor', 'inhib', 'get_functions', '_bitlist_from_value', '_is_bit_list', 'from_str', 'const'}
    tracer = Tracer(skip, collapse)

    from circuits.examples.keccak import Keccak
    from circuits.neurons.core import Bit
    from circuits.utils.format import Bits
    def test(message: Bits, k: Keccak) -> list[Bit]:
        hashed = k.digest(message)
        return hashed.bitlist
    k = Keccak(c=10, l=0, n=1, pad_char='_')
    phrase = "Reify semantics as referentless embeddings"
    message = k.format(phrase, clip=True)
    trace = tracer.run(test, message=message, k=k)

    out_levels = get_output_levels(trace.root)
    print(len(out_levels))
    for level in out_levels:
        print(len(Bits(list(level))))
    
    hide = {'gate'}
    print(trace.root.__str__(hide=hide))

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




# @dataclass
# class Source:
#     node: CallNode | None  # None = in inputs
#     height: int  # how high the source is in the
#     index: int  # node.outputs[index] == signal

# @dataclass
# class CopyTower:
#     tower: list[Source | None]
#     node: CallNode  # node in which this copy tower exists
#     highest: int  # highest copy in the tower (not counting disconnected copies)
#     lowest: int  # lowest copy in the tower
#     initial_state: str = ""

#     @classmethod
#     def from_source(cls, node: CallNode, source: Source) -> 'CopyTower':
#         tower: list[Source | None] = [None] * (node.h+1)
#         assert len(tower) > source.height, f"Failed! source.height={source.height}, len(tower)={len(tower)} \n{node}"
#         tower[source.height] = source  # Set the source at its height
#         assert source is not None
#         assert tower[source.height] is not None 
#         initial_state = f"{[(s.node, source.height) if s is not None else None for s in tower]}"
#         # assert initial_state != "[None, None]"
#         return cls(tower, node, source.height, source.height, initial_state)

#     def get_source(self, height: int, count: int) -> Source:
#         """Gets the source at the given height, adding copies if necessary.
#         E.g. if node at height=3 needs source, it will ask get_source(3)"""
#         assert len(self.tower) > height, f"Failed! height={height}, len(tower)={len(self.tower)}, \n{self.node}"
#         assert self.lowest is not None
#         assert height >= self.lowest, f"Sink before source! {height}<{self.lowest}; \n{self.node}"
#         if self.tower[height] is None:
#             self.add_copies(height, count)
#         result = self.tower[height]
#         assert result is not None, f"No source found at height {height} in tower - add_copies failed."
#         return result
#         # TODO: give warning if sink below any source
    
#     def add_copies(self, height: int, count: int) -> None:
#         """Adds copies up to given height"""
#         # TODO: update count
#         # initial_tower = f"tower:{[s.node if s is not None else None for s in self.tower]}"
#         while self.highest < height:
#             # print(f'added copy to tower at height {self.node.name}-{self.node.count}, height={height}, highest={self.highest}, lowest={self.lowest}')
#             self.highest += 1
#             if self.tower[self.highest] is not None:
#                 continue  # already has a copy at this height
#             new_left = self.node.add(bot=self.highest-1, top=self.highest, width=1)  # make room for a copy
#             copy_node = CallNode(name='copy', count=count, parent=self.node, depth=self.node.depth + 1, bot=self.highest-1, top=self.highest, left=new_left, right=new_left+1)
#             # print(f"Connecting sink to source at {self.node.name}")
#             self.node.children.append(copy_node)  # Add copy node to the children
#             self.tower[self.highest] = Source(node=copy_node, height=self.highest, index=0)
#         string = f"Failed copies: height={height}, len(tower)={len(self.tower)}"
#         string += f", highest={self.highest}, lowest={self.lowest}, tower:{[s.node if s is not None else None for s in self.tower]}"
#         string += f", initial_tower: {self.initial_state},"
#         string += f"\n{self.node}"
#         assert self.tower[height] is not None, string



# def set_connections(node: CallNode) -> None:
#     # problem: some inputs might from layers further than the previous layer
#     # idea: connect locations where the signal is required (sinks) and produced (sources)
#     # sources = inputs and children outputs
#     # sinks = outputs and children inputs
#     # method: copy a signal from its earlierst source to latest sink
#     if node.name != 'round':
#         return

#     if node.h == 0 or node.name in {'gate', 'copy', 'const'}:
#         # TODO: consider connecting inputs to outputs
#         return
#     # ignore gate subcalls
#     if 'gate' in [p.name for p in node.fpath[:-1]]:
#         return

#     # collect sources
#     sources: dict[Signal, CopyTower] = dict()
#     for i, inp in enumerate([s for s,_ in node.inputs]):
#         if inp not in sources:
#             sources[inp] = CopyTower.from_source(node, Source(node=None, height=0, index=i))
#     for c in node.children:
#         for i, out in enumerate([s for s,_ in c.outputs]):
#             if out not in sources:
#                 sources[out] = CopyTower.from_source(node, Source(node=c, height=c.top, index=i))
#             elif c.top < sources[out].highest:  # available earlier
#                 sources[out].highest = c.top

#     # collect sinks
#     copy_counter = 0
#     for i, sink in enumerate([s for s,_ in node.outputs]):
#         assert sink in sources, f"Sink {sink} not found in sources. This should not happen."
#         source = sources[sink].get_source(height=node.h, count=copy_counter)
#         # print(f"Connecting sink {node.name}-outputs to source {source.node.name} at index {source.index}")
#         copy_counter += 1
#         node.input_sources.append((source.node, source.index))  # Add the source node and its index to the input sources
#     for c in node.children:
#         for sink in [s for s,_ in c.inputs]:
#             assert sink in sources, f"Sink {sink} not found in sources. This should not happen."
#             source = sources[sink].get_source(height=c.bot, count=copy_counter)
#             # print(f"Connecting sink {c.name}-inputs to source at index {source.index}")
#             copy_counter += 1
#             c.input_sources.append((source.node, source.index))  # Add the source node and its index to the input sources
#     # print(f"Connecting {node.name}")

#     # TODO: use ordered sets for determinism
#     # TODO: consider optimizations, e.g. use sets, or make fewer copies

#     # children_by_descending_out_height = sorted(node.children, key=lambda c: c.top, reverse=True)
#     # sorted_children = sorted(node.children, key=lambda c: c.top, reverse=True)
#     # out_sources_in_descending_height = [{c.top: c.output_set} for c in sorted_children]
#     # out_sources_in_descending_height += [{0: node.input_set}]  # inputs can also be used in outputs
#     # for output in [s for s, _ in node.outputs]:
