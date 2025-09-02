from dataclasses import dataclass, field
from collections.abc import Callable, Generator
from typing import Literal, Any

from circuits.utils.misc import OrderedSet
from circuits.compile.monitor import CallNode
from circuits.neurons.core import Bit
from circuits.compile.graph import Origin


@dataclass(eq=False)
class Flow:
    """
    Represents data flow between blocks
    Same data instance can occur at many positions, e.g. due to copying
    """
    data: Bit
    block: 'Block'
    direction: Literal['in', 'out']
    indices: list[int] = field(default_factory=list[int])
    flat_index: int = 0  # Flattened index
    creator: 'Block | None' = None
    prev: 'Flow | None' = None  # Previous flow on the same depth

    def __post_init__(self):
        assert isinstance(self.data, Bit), f"Flow {self.block.path} has a non-bit data: {type(self.data)} {self.data}"

    @property
    def path(self) -> str:
        history: list['Flow'] = []
        anc = self
        while anc is not None:
            history.append(anc)
            anc = anc.prev
        splits = [anc.block.path.split('.') for anc in history]
        nestings = [len(s) for s in splits]
        ascent_end = nestings.index(min(nestings))
        core = '.'.join(splits[0][:len(splits[0])-ascent_end-1])
        res = f"{core}: "
        if len(splits)>1 and splits[0]>=splits[1]:
            ascent = ""
            for i in range(ascent_end+1):
                ascent = f"{splits[i][-1]}[{history[i].flat_index}]." + ascent
            res += f"{ascent[:-1]}"
        descent_len = len(history)-ascent_end-1
        if descent_len > 0:
            descent = ""
            for i in range(ascent_end+1, ascent_end+1+descent_len):
                descent += f".{splits[i][-1]}[{history[i].flat_index}]"
            res += f" \tfrom\t {descent[1:]}"
        if history[-1].block.flavour == 'copy' and history[-1].prev is None:
            assert history[-1].block.original is not None
            res += f"\t original: {history[-1].block.original.path}"
        return res


@dataclass(eq=False)
class Block:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    path: str
    inputs: OrderedSet[Flow] = field(default_factory=OrderedSet[Flow])
    outputs: OrderedSet[Flow] = field(default_factory=OrderedSet[Flow])
    parent: 'Block | None' = None
    children: list['Block'] = field(default_factory=list['Block'])
    flavour: Literal['function', 'creator', 'copy', 'input', 'output', 'untraced'] = 'function'
    is_creator: bool = False

    created: OrderedSet[Bit] = field(default_factory=OrderedSet[Bit])
    consumed: OrderedSet[Bit] = field(default_factory=OrderedSet[Bit])

    # Positioning relative to parent's bottom/left edge
    bot: int = 0  # Bottom depth 
    top: int = 0  # Top depth
    left: int = 0  # left index
    right: int = 0  # right index

    # Absolute positioning - relative to roots's bottom/left edge
    abs_x: int = 0  # Absolute index coordinate (leftmost edge)
    abs_y: int = 0  # Absolute depth (bottom edge)
    levels: list[int] = field(default_factory=list[int])  # level widths of the node in the call tree

    # For visualising color and block output
    formatter: Callable[[Bit], str] = lambda x: str(x)  # Function to format tracked instances
    out_str: str = ""  # String representation of the outputs
    outdiff: str = ""  # String representation of the outputs that differ from some other node
    tags: set[str] = field(default_factory=set[str])
    nesting: int = 0  # Nesting level of the block in the call tree
    max_leaf_nesting: int = -1
    original: 'Block | None' = None  # original creator of copy

    info: dict[str, Any] = field(default_factory=dict[str, Any])  # for storing additional info
    
    origin: Origin = Origin(0, (), 0)


    @property
    def path_from_root(self) -> tuple['Block', ...]:
        """Returns the function path as a tuple of Block from root to this node."""
        path: list['Block'] = []
        current: Block | None = self
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
        return self.right - self.left

    @property
    def creation(self) -> Flow:
        assert self.is_creator and len(self.outputs)==1
        return list(self.outputs)[0]

    def update_levels(self, bot: int, top: int, width: int) -> int:
        """Adds a child node at bot-top depth. width = child width.
        Updates self.levels widths. Returns the new child left index"""
        if len(self.levels) < top:
            self.levels.extend(0 for _ in range(len(self.levels), top))
        depths = list(range(bot, top))
        widths = [self.levels[h] for h in depths]
        new_left = max(widths) if widths else 0  # Find the maximum width at the depths
        self.levels[bot:top] = [new_left + width] * len(depths)  # Update all levels in the range to child right
        return new_left

    def __repr__(self):
        return f"{self.path}"

    def full_info(self) -> str:
        s = f"name: {self.name}\n"
        s += f"path: {self.path}\n"
        s += f"io: ({len(self.inputs)}→{len(self.outputs)})\n"
        s += f"nesting level: {self.nesting}\n"
        s += f"x: {self.abs_x}, y: {self.abs_y}, w: {self.w}, h: {self.h}\n"
        if self.original:
            s += f"original: {self.original.path}\n"
        if self.tags:
            s += f"tags: {self.tags}\n"
        if len(self.out_str) > 50:
            out_str = self.out_str[:50] + "..."
            outdiff = self.outdiff[:50] + "..."
        else:
            out_str = self.out_str
            outdiff = self.outdiff
        s += f"out_str: '{out_str}'\n"
        if self.outdiff:
            s += f"outdiff: '{outdiff}'\n"
        return s

    @classmethod
    def from_root_node(cls, root_node: CallNode[Bit]) -> 'Block':
        def walk_nodes(node: CallNode[Bit]) -> Generator[CallNode[Bit], None, None]:
            yield node
            for c in node.children:
                yield from walk_nodes(c)

        node_to_block: dict[CallNode[Bit], Block] = {}
        for n in walk_nodes(root_node):

            # Get path
            path = ""
            if n.parent is not None:
                path = f"{node_to_block[n.parent].path}"
                if n.parent.parent is not None:
                    path += "."
                path += f"{n.name}" 
                if n.parent.counts[n.name] > 1:  # exclude count if function is only called once
                    path += f"-{n.count}"

            # Create block
            b = cls(n.name, path)
            b.inputs = OrderedSet([Flow(inp, b, 'in', indices, i)
                for i, (inp, indices) in enumerate(n.inputs)])
            b.outputs = OrderedSet([Flow(out, b, 'out', indices, i)
                for i, (out, indices) in enumerate(n.outputs)])
            node_to_block[n] = b
            # if b.inputs:
            #     print(f"b.inputs: {b.inputs}")

            # Mark gates
            if n.name == 'gate':
                # assert n.creation is not None, f"gate {b.path} has no creation"
                b.outputs = OrderedSet([Flow(list(n.outputs)[0][0], b, 'out')])
                b.flavour = 'creator'
                b.is_creator = True

            # Add parent
            if n.parent and n.parent.name != 'gate':  # not tracking gate subcalls
                b.parent = node_to_block[n.parent]
                b.parent.children.append(b)

        root = node_to_block[root_node]
        root.name = "root"
        root.path = "root"
        return root



def get_lca_children_split(x: Block, y: Block) -> tuple[Block, Block]:
    """
    Find the last common ancestor of x and y.
    Then returns its two children a and b that are on paths to x and y respectively.
    """
    x_path = x.path_from_root
    y_path = y.path_from_root
    for i in range(min(len(x_path), len(y_path))):
        if x_path[i] != y_path[i]:
            return x_path[i], y_path[i]  # Found the first mismatch, return lca_child_to_x, lca_child_to_y
    raise ValueError(f"b and its ancestor are on the same path to root: b ancestor={x.path}, creator ancestor={y.path}")


def update_ancestor_depths(b: Block) -> None:
    """On return of a block b, set its depth to be after its inputs, update ancestor depths if necessary"""
    for inflow in b.inputs:
        if inflow.creator is None:
            continue
        b_ancestor, creator_ancestor = get_lca_children_split(b, inflow.creator)
        if b_ancestor.bot < creator_ancestor.top:  # current block must be above the parent block
            h_change = creator_ancestor.top - b_ancestor.bot
            b_ancestor.bot += h_change
            b_ancestor.top += h_change


def set_left_right(b: Block) -> None:
    """
    Sets the left and right position of the block based on its parent
    Assumes that bot/top are set to the correct values
    """
    w = max(b.levels) if b.levels else b.w  # current_block_width
    if len(b.outputs) > w:
        w = len(b.outputs)
    if not b.parent:
        index_shift = 0
    else:
        index_shift = b.parent.update_levels(b.bot, b.top, w)
    b.left += index_shift
    b.right = b.left + w


def traverse(b: Block, order: Literal['call', 'return'] = 'call') -> Generator[Block, None, None]:
    """Walks the call tree and yields each node."""
    if order == 'call':
        yield b
    for child in b.children:
        yield from traverse(child, order)
    if order == 'return':
        yield b


def add_copies_to_block(b: Block) -> None:
    """
    Ensures that within a block its outputs and its children inputs are available
    in this block at the same depth as their creators
    """
    if b.is_creator:
        return  # creator blocks do not need copies inside

    required: dict[int, OrderedSet[Flow]] = {d: OrderedSet() for d in range(b.h+1)}
    available: dict[int, OrderedSet[Flow]] = {d: OrderedSet() for d in range(b.h+1)}
    required[b.h] = b.outputs
    available[0] = b.inputs
    for c in b.children:
        required[c.bot] |= c.inputs
        available[c.top] |= c.outputs
        if c.is_creator:
            available[c.top].add(c.creation)

    # descend from top to bot
    n_copies = 0
    copies: list[Block] = []
    for d in reversed(range(b.h + 1)):
        available_data = {inst.data: inst for inst in available[d]}
        for req in required[d]:
            if req.data not in available_data:

                if d==0:
                    b.tags.add('missing')
                    raise ValueError(f"{req.creator.path if req.creator else 'unknown'} not available at {b.path} inputs")

                # create a copy
                copy = Block("copy", b.path+".copy", is_creator=True, parent=b, flavour='copy', tags={'copy'})
                copies.append(copy)
                outflow = Flow(req.data, copy, 'out', creator=copy, prev=None)  # no prev
                inflow = Flow(req.data, copy, 'in', creator=req.creator)  # prev to be set later, creator maybe
                copy.outputs.add(outflow)
                copy.inputs.add(inflow)
                copy.original = req.block.original if req.block.original is not None else req.creator
                copy.path += f"-{n_copies}"
                n_copies += 1

                available_data.update({req.data: outflow})
                required[d-1].add(inflow)

            avail = available_data[req.data]
            req.prev = avail
            req.creator = avail.creator

    # reverse order of copies to ensure that they are created after their creators
    for copy in reversed(copies):
        b.children.append(copy)


def add_copy_blocks(root: Block) -> None:
    for b in traverse(root, 'return'):
        add_copies_to_block(b)
    # propagate .creator:
    for b in traverse(root, 'call'):
        for inp in b.inputs:
            if inp.prev is not None:
                inp.creator = inp.prev.creator


def set_flow_creator_for_io_of_each_block(root: Block) -> None:
    """Sets the creator of each flow to the block that created it"""
    # record all instance creators
    bit_to_block: dict[Bit, Block] = {}
    for b in traverse(root):
        if b.is_creator:
            bit_to_block[b.creation.data] = b

    # set creator of each flow
    for b in traverse(root, 'return'):        
        for flow in b.inputs | b.outputs:
            if flow.creator is None:
                if flow.data not in bit_to_block:
                    print(f"This block has io created outside of the tree: {b.path} at index {flow.flat_index}")
                    b.tags.add('missing_io')
                else:
                    flow.creator = bit_to_block[flow.data]



def get_lca(blocks: list[Block]) -> Block:
    """Returns the lowest common ancestor of the blocks"""
    assert len(blocks)>1
    paths_from_root = [b.path_from_root for b in blocks]
    i = 0
    while True:
        try:
            ancestor = paths_from_root[0][i]
            for path in paths_from_root[1:]:
                if path[i] != ancestor:
                    break
            i += 1
        except:
            break
    first_mismatch_idx = i
    lca = paths_from_root[0][first_mismatch_idx-1]
    return lca


def assign_inputs(root: Block) -> None:
    """Assigns inputs to blocks based on created and consumed bits"""
    for b in traverse(root, 'return'):
        for c in b.children:
            b.created |= c.created
            b.consumed |= c.consumed
        if b.is_creator:
            b.created |= OrderedSet([list(b.outputs)[0].data])
            b.consumed |= OrderedSet(list(b.outputs)[0].data.source.incoming)
        else:
            b.consumed |= OrderedSet([out.data for out in b.outputs])
    for b in traverse(root, 'call'):
        inp_bits = b.consumed - b.created
        if b.path != "root":
            b.inputs = OrderedSet([Flow(bit, b, 'in') for bit in inp_bits])


def add_input_blocks(root: Block) -> None:
    input_blocks: list[Block] = []
    for j, flow in enumerate(root.inputs):
        b = Block('input', f'input-{j}', is_creator=True, flavour='input', tags={'input'}, abs_x=j)
        outflow = Flow(flow.data, b, 'out', prev=None)
        b.outputs = OrderedSet([outflow])
        b.parent = root
        input_blocks.append(b)
        root.inputs = OrderedSet()  # remove inputs from root
    root.children = input_blocks + root.children  # add input blocks to the front
    for b in traverse(root):
        if b.parent  and b.parent.name == 'root' and b.flavour and b.bot==0 and b.flavour != 'input': 
            b.bot += 1
            b.top += 1


def add_output_blocks(root: Block) -> None:
    for j, root_outflow in enumerate(root.outputs):
        assert root_outflow.creator is not None  # this should be set by set_flow_creator_for_io_of_each_block
        b = Block('output', f'output-{j}', is_creator=True, flavour='output', tags={'output'}, abs_x=j)
        outflow = Flow(root_outflow.data, b, 'out', creator = b, prev=None)
        inflow = Flow(root_outflow.data, b, 'in', creator=root_outflow.creator, prev=root_outflow.prev)
        root_outflow.creator = b
        root_outflow.prev = outflow
        b.outputs = OrderedSet([outflow])
        b.inputs = OrderedSet([inflow])
        b.parent = root
        root.children.append(b)


def fold_untraced_bits(root: Block) -> None:
    """Finds bits not traced by ftrace and folds them into gates consuming them"""

    # find bits with known creators
    traced_bits: OrderedSet[Bit] = OrderedSet()
    bit_to_block: dict[Bit, Block] = dict()
    for b in traverse(root, 'return'):
        if b.is_creator:
            gate_bit = b.creation.data
            assert isinstance(gate_bit, Bit), f"gate {b.path} has a non-bit creation: {type(gate_bit)} {gate_bit}, creation={b.creation}"
            traced_bits.add(gate_bit)
            bit_to_block[gate_bit] = b

    # backwards scan from gates to find untraced bits
    untraced_bits: OrderedSet[Bit] = OrderedSet()
    frontier: OrderedSet[Bit] = OrderedSet()
    frontier |= traced_bits
    while frontier:
        new_frontier: OrderedSet[Bit] = OrderedSet()
        for bit in frontier:
            for parent in bit.source.incoming:
                if parent not in traced_bits and parent not in untraced_bits:
                    untraced_bits.add(parent)
                    new_frontier.add(parent)
        frontier = new_frontier

    # ensure that untraced bits are constant
    input_bits: OrderedSet[Bit] = OrderedSet()
    for b in traverse(root):
        if b.flavour == 'input':
            input_bits.add(b.creation.data)
    live_untraced_bits: OrderedSet[Bit] = OrderedSet()
    frontier |= untraced_bits
    while frontier:
        new_frontier = OrderedSet()
        for bit in frontier:
            for parent in bit.source.incoming:
                if parent in input_bits:
                    live_untraced_bits.add(bit)
        frontier = new_frontier
    assert len(live_untraced_bits) == 0, "Live untraced bits are currently unsupported"

    # fold untraced bits into gate biases
    for b in traverse(root, 'call'):
        inflows = b.inputs
        for j, inflow in enumerate(list(inflows)):
            if inflow.data in untraced_bits:

                # fold untraced bit into gate bias
                if b.name == 'gate':
                    untraced_w = b.creation.data.source.weights[j]
                    untraced_value = b.creation.data.source.incoming[j].activation
                    b.origin = Origin(0, (), int(untraced_value * untraced_w))
                    b.tags.add('untraced')

                # remove from inputs
                b.inputs.remove(inflow)

        outflows = b.outputs
        for outflow in list(outflows):
            if outflow.data in untraced_bits:
                b.outputs.remove(outflow)



def set_layout(root: Block) -> Block:
    """Sets the coordinates for the blocks in the call tree"""
    for b in traverse(root):
        # Reset if set_layout was already called
        b.bot = 0
        b.top = 0
        b.left = 0
        b.right = 0
        b.levels = []
        b.abs_x = 0
        b.abs_y = 0
        b.max_leaf_nesting = -1
        # TODO: refactor to not need resetting

    # inp_blocks = create_input_blocks(root)
    set_flow_creator_for_io_of_each_block(root)
    for b in traverse(root, order='return'):

        # Set creator/copy size to 1x1
        if b.is_creator:
            b.top = b.bot + 1
            b.right = b.left + 1
        if b.parent and b.parent.name == 'root' and b.flavour and b.bot==0 and b.flavour != 'input': 
            b.bot += 1
            b.top += 1

        # Ensure b comes after its inputs are created
        update_ancestor_depths(b)

        # Ensure correct top depth
        if b.children:
            b.top = b.bot + max([c.top for c in b.children])

        set_left_right(b)

    # Now that .left and .bot are finalized, set absolute coordinates
    for b in traverse(root):
        if b.parent is not None:
            b.abs_x = b.left + b.parent.abs_x
            b.abs_y = b.bot + b.parent.abs_y

    return root



def delete_zero_h_blocks(root: Block) -> None:
    for b in traverse(root):
        b.children = [c for c in b.children if c.h != 0]


def set_formatting_info(root: Block) -> None:
    # set nesting depth info
    for b in traverse(root):
        b.nesting = b.parent.nesting + 1 if b.parent else 0
    for b in traverse(root, 'return'):
        b.max_leaf_nesting = max([c.max_leaf_nesting for c in b.children])+1 if b.children else 0

        # set output string
        b.out_str = "".join([str(int(out.data.activation)) for out in b.outputs])
        
        # set live/constant tags
        if b.flavour == 'input':
            b.tags.add('live')
        for inflow in b.inputs:
            assert inflow.creator is not None
            if inflow.creator.flavour == 'input' or 'live' in inflow.creator.tags:
                b.tags.add('live')
            if inflow.creator.flavour == 'copy':
                assert inflow.creator.original is not None
                if 'live' in inflow.creator.original.tags:
                    b.tags.add('live')
    for b in traverse(root):
        if not 'live' in b.tags:
            b.tags.add('constant')
        b.tags.discard('live')
    root.tags.discard('constant')


def mark_differences(root1: Block, root2: Block) -> None:
    """Highlights the differences between two block trees"""
    for b1, b2 in zip(traverse(root1), traverse(root2)):
        assert b1.path == b2.path, f"Block paths do not match: {b1.path} != {b2.path}"
        if b1.out_str != b2.out_str:
            b1.tags.add('different')
            b2.tags.add('different')
            for out1, out2 in zip(b1.out_str, b2.out_str):
                diff = ' ' if out1==out2 else out2
                b1.outdiff += diff
                b2.outdiff += diff


from circuits.compile.monitor import Tracer, find
@dataclass
class BlockTracer:
    collapse: set[str] = field(default_factory=set[str])
    use_defaults: bool = False

    def __post_init__(self):
        if self.use_defaults:
            c = {'__init__', '__post_init__', '<lambda>'}
            c |= {'outgoing', 'const', 'xor', 'inhib', 'step'}
            c |= {'format', 'bitlist', '_bitlist_from_value', '_is_bit_list', 'from_str'}
            c |= {'_bitlist_to_msg', 'msg_to_state', 'get_round_constants', 'get_functions'}
            c |= {'lanes_to_state', 'state_to_lanes', 'get_empty_lanes', 'copy_lanes'}
            c |= {'rho_pi', 'rot', 'reverse_bytes'}
            self.collapse |= c

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Block:
        tracer = Tracer[Bit](Bit, self.collapse)
        with tracer.trace():
            out = func(*args, **kwargs)
        tracer.root.inputs = find(args + tuple(kwargs.values()), Bit)
        tracer.root.outputs = find(out, Bit)
        r = Block.from_root_node(tracer.root)
        assign_inputs(r)
        add_input_blocks(r)
        fold_untraced_bits(r)
        set_layout(r)
        add_output_blocks(r)
        set_layout(r)
        delete_zero_h_blocks(r)
        add_copy_blocks(r)
        set_layout(r)
        set_formatting_info(r)
        return r
