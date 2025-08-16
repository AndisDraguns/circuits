import sys
from types import FrameType
from dataclasses import dataclass, field
from typing import Any, Literal

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
    is_live: bool = False  # live = generates signals that are downstream from inputs
    depth: int = 0  # Nesting depth in the call tree
    bot: int = 0  # Bottom height of the node in the call tree (relative to parent.top)
    top: int = 0  # Top height of the node in the call tree (relative to self.bot)
    left: int = 0  # left position of the node in the call tree (relative to parent.left)
    right: int = 0  # right position of the node in the call tree (relative to self.left)
    levels: list[int] = field(default_factory=list[int])  # level widths of the node in the call tree
    inputs: OrderedSet[Signal] = field(default_factory=OrderedSet[Signal])
    outputs: OrderedSet[Signal] = field(default_factory=OrderedSet[Signal])
    children: list['CallNode'] = field(default_factory=list['CallNode'])
    fn_counts: dict[str, int] = field(default_factory=dict[str, int])  # child fn name -> # direct calls in self

    skip: bool = False

    # post-processing info:
    x: int = 0  # Absolute x coordinate (leftmost edge)
    y: int = 0  # Absolute y coordinate (bottom edge)
    full_name: str | None = None  # Full name of the node in the call tree
    out_str: str = ""  # String representation of the outputs
    highlight: bool = False  # Whether to highlight this node
    outdiff: str = ""

    # connections info
    # input_sources: list[tuple['CallNode|None', int]] = field(default_factory=list[tuple['CallNode|None', int]])  # (source node, its output index) for each input

    def create_child(self, fn_name: str) -> 'CallNode':
        self.fn_counts[fn_name] = self.fn_counts.get(fn_name, -1) + 1
        child = CallNode(name=fn_name, count=self.fn_counts[fn_name], parent=self, depth=self.depth+1)
        self.children.append(child)
        return child

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
        return ""
        indent = "  " * level
        info = self.info_str()
        child_names = "".join(f"\n{c.__str__(level + 1, hide)}" for c in self.children if c.name not in hide)
        res = f"{indent}{info}{child_names}"
        return res

    def __repr__(self):
        return f"n {self.name}"

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
    For each input, its creator gate node g is located. 
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
    assert len(g.outputs) == 1
    if len(g.outputs) == 0:
        return
    s = list(g.outputs)[0]  # Get the output signal of the gate
    assert g.name == 'gate', f"Expected gate node, got {g.name}"
    assert len(g.outputs) == 1
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
    # input_args: Any
    # input_kwargs: Any
    output: Any = None
    max_depth: int = 0

    def highlight_differences(self, other: 'Trace') -> None:
        """
        Highlights the differences between two call trees.
        Sets 'highlight' flag in for each call node that differs from the corresponding node in the other tree.
        """
        gen1 = walk_generator(other.root)
        gen2 = walk_generator(self.root)
        for val1, val2 in zip(gen1, gen2):
            node1, node2 = val1[0], val2[0]
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
    use_defaults: bool = False

    def __post_init__(self) -> None:
        self.skip |= {'set_trace'}  # no need to track set_trace
        self.collapse |= {'<genexpr>'}  # avoids handling generator interactions with stack
        c = {'__init__', '__post_init__', '<lambda>', '<genexpr>'}
        c |= {'outgoing', 'const', 'xor', 'inhib', 'step'}
        c |= {'format', 'bitlist', '_bitlist_from_value', '_is_bit_list', 'from_str'}
        c |= {'_bitlist_to_msg', 'msg_to_state', 'get_round_constants', 'get_functions'}
        c |= {'lanes_to_state', 'state_to_lanes', 'get_empty_lanes', 'copy_lanes'}
        c |= {'rho_pi', 'rot', 'reverse_bytes'}
        if self.use_defaults:
            self.collapse |= c

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> CallNode:
        def root_wrapper_fn(*args: Any, **kwargs: Any) -> Any:
            """Wraps a function call to avoid special handling of the root call"""
            return func(*args, **kwargs)
        trace = self.run_fn(root_wrapper_fn, *args, **kwargs)
        # trace = self.post_process_trace(trace)
        return trace

    def run_fn(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> CallNode:
        """
        Execute function while building a tree of CallNodes tracking Signal flow.
        Args:
            func: Function to trace
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
        """
        root_wrapper = CallNode(name=func.__name__, depth=-1)
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
                if fn_name in self.collapse:
                    return trace_handler  # Continue tracing children, but don't create a node

                # Create a new node
                parent = stack[-1]
                node = parent.create_child(fn_name)
                stack.append(node)

                # Record inputs
                input_objects = [value for _, value in frame.f_locals.items()]
                node.inputs = find_signals(input_objects)


            elif event == 'return':
                fn_name = frame.f_code.co_name

                # Handle unrecorded returns
                if fn_name == root_wrapper.name:
                    return trace_handler
                if id(frame) == skipped_frame_id:
                    skipping = False
                    return trace_handler
                if skipping:
                    assert not fn_name == 'gate', "Skipped gate call"
                    return trace_handler
                if fn_name in self.collapse:
                    return trace_handler

                # Record outputs
                node = stack.pop()
                node.outputs = find_signals(arg)

            return trace_handler

        # Execute with tracing
        original_trace = sys.gettrace()
        try:
            set_trace(trace_handler)
            # result = func(*args, **kwargs)
            _ = func(*args, **kwargs)
        finally:
            set_trace(original_trace)

        return root_wrapper
        # root = root_wrapper.children[0]
        # root.parent = None
        # return Trace(root_wrapper, result)
        # root = root_wrapper.children[0]
        # root.parent = None
        # return Trace(root, result)


    def post_process_trace(self, trace: Trace) -> Trace:
        """Processes the call tree"""
        for n, _ in walk_generator(trace.root, order='return'):
            
            # Set full name
            if n.parent is None:  # is root
                n.full_name = f"{n.name}-{n.count}"
            else:
                n.full_name = f"{n.parent.full_name}.{n.name}-{n.count}"

            if n.depth > trace.max_depth:
                trace.max_depth = n.depth

            n.out_str = Bits(list(n.outputs)).bitstr

            # The following must be on return
            # Sets node's coordinates
            if n.name == 'gate':
                process_gate_return(n)
            update_ancestor_heights(n)
            set_top(n)
            set_left_right(n)

            if any([c.is_live for c in n.children]):
                n.is_live = True

        # Now that .left and .bot are finalized, we can  set absolute coordinates
        for n, _ in walk_generator(trace.root):
            n.set_absolute_coordinates()

        return trace


def walk_generator(node: CallNode, order: Literal['call', 'return', 'both', 'either'] = 'either'
                   ) -> Generator[tuple[CallNode, Literal['call', 'return']], None, None]:
    """Walks the call tree and yields each node."""
    if order in {'call', 'both', 'either'}:
        yield node, 'call'
    for child in node.children:
        yield from walk_generator(child, order)
    if order in {'return', 'both'}:
        yield node, 'return'


def get_output_levels(root: CallNode) -> list[OrderedSet[Signal]]:
    """Gets the output levels of the call tree."""
    levels: list[OrderedSet[Signal]] = [OrderedSet() for _ in range(root.top+1)]
    for n, _ in walk_generator(root):
        for out in n.outputs:
            levels[n.top].add(out)
    return levels


def node_walk_generator(node: CallNode, order: Literal['call', 'return', 'both', 'either'] = 'either'
                   ) -> Generator[tuple[CallNode, Literal['call', 'return']], None, None]:
    """Walks the call tree and yields each node."""
    if order in {'call', 'both', 'either'}:
        yield node, 'call'
    for child in node.children:
        yield from walk_generator(child, order)
    if order in {'return', 'both'}:
        yield node, 'return'


if __name__ == '__main__':
    tracer = Tracer(use_defaults=True)

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
    hide = {'gate'}
    # print(trace.root.__str__(hide=hide))



    # out_levels = get_output_levels(trace.root)
    # print(len(out_levels))
    # for level in out_levels:
    #     print(len(Bits(list(level))))

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


                # if func_name != 'gate':
                #     print(f"{len(stack)*' '}i {func_name}")

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
