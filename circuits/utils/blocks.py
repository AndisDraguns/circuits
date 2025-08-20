from dataclasses import dataclass, field
from collections.abc import Callable, Generator
from typing import Literal

from circuits.utils.misc import OrderedSet
from circuits.utils.ftrace import CallNode


@dataclass(eq=False)
class Flow[T]:
    """Represents data flow between blocks"""
    # Same data instance can occur at many positions, e.g. due to copying
    data: T
    depth: int = -1  # Absolute vertical position
    index: int = -1  # Absolute horizontal position
    creator: 'Block[T] | None' = None
    prev_copy: 'Block[T] | None' = None
    prev: 'Flow[T] | None' = None  # Previous flow on the same depth


@dataclass(eq=False)
class Block[T]:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    path: str
    inputs: OrderedSet[Flow[T]] = field(default_factory=OrderedSet[Flow[T]])
    outputs: OrderedSet[Flow[T]] = field(default_factory=OrderedSet[Flow[T]])
    parent: 'Block[T] | None' = None
    children: list['Block[T]'] = field(default_factory=list['Block[T]'])
    creation: Flow[T] | None = None  # T created by this node, if any
    flavour: Literal['function', 'creator', 'copy', 'input'] = 'function'

    # Positioning
    # Relative to parent's bottom/left edge:
    bot: int = 0  # Bottom depth 
    top: int = 0  # Top depth
    left: int = 0  # left index
    right: int = 0  # right index
    # Absolute = relative to roots's bottom/left edge:
    abs_x: int = 0  # Absolute index coordinate (leftmost edge)
    abs_y: int = 0  # Absolute depth (bottom edge)
    levels: list[int] = field(default_factory=list[int])  # level widths of the node in the call tree

    # Copying
    copy_levels: dict[int, list['Block[T]']] = field(default_factory=dict[int, list['Block[T]']])  # level -> blocks to copy

    # For visualizng color and block output
    formatter: Callable[[T], str] = lambda x: str(x)  # Function to format tracked instances
    out_str: str = ""  # String representation of the outputs
    outdiff: str = ""  # String representation of the outputs that differ from some other node
    tags: set[str] = field(default_factory=lambda: {'constant'})
    nesting: int = 0  # Nesting level of the block in the call tree
    max_leaf_nesting: int = -1


    @property
    def path_from_root(self) -> tuple['Block[T]', ...]:
        """Returns the function path as a tuple of Block from root to this node."""
        path: list['Block[T]'] = []
        current: Block[T] | None = self
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
    def unwrapped(self) -> 'Block[T]':
        assert self.name == "root_wrapper_fn"
        b = self.children[0]  # get rid of the root wrapper fn
        b.parent = None
        return b

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

    # def info_str(self) -> str:
    #     """Returns a string representation of the node's info, excluding its children"""
    #     call_name = f"{self.name}"
    #     io = ""
    #     if self.inputs or self.outputs:
    #         io = f"({len(self.inputs)}→{len(self.outputs)})"
    #     bot_top = f"[b={self.bot}..t={self.top}]"
    #     left_right = f"[l={self.left}..r={self.right}]"
    #     res = f"{call_name} {io} {bot_top} {left_right}"
    #     return res

    # def __str__(self, level: int = 0, hide: set[str] = set()) -> str:
    #     indent = "  " * level
    #     info = self.info_str()
    #     child_names = "".join(f"\n{c.__str__(level + 1, hide)}" for c in self.children if c.name not in hide)
    #     res = f"{indent}{info}{child_names}"
    #     return res
    
    def __repr__(self):
        return f"{self.name}"

    def full_info(self) -> str:
        s = f"name-count: {self.name}\n"
        s += f"io: ({len(self.inputs)}→{len(self.outputs)})\n"
        s += f"path: {self.path}\n"
        s += f"nesting level: {self.nesting}\n"
        s += f"x: {self.abs_x}, y: {self.abs_y}, w: {self.w}, h: {self.h}\n"
        s += f"tags: {self.tags}\n"
        s += f"out_str: '{self.out_str}'\n"
        if self.outdiff:
            s += f"outdiff: '{self.outdiff}'\n"
        return s

    @classmethod
    def from_root_node(cls, root_node: CallNode[T]) -> 'Block[T]':
        def walk_nodes(node: CallNode[T]) -> Generator[CallNode[T], None, None]:
            yield node
            for c in node.children:
                yield from walk_nodes(c)

        to_block: dict[CallNode[T], Block[T]] = {}
        for n in walk_nodes(root_node):

            # Get path
            path = ""
            if n.parent is not None:
                path = f"{to_block[n.parent].path}"
                if n.parent.parent is not None:
                    path += "."
                path += f"{n.name}" 
                if n.parent.fn_counts[n.name] > 1:  # exclude count if function is only called once
                    path += f"-{n.count}"

            # Get inputs and outputs
            inputs = OrderedSet([Flow[T](inp) for inp in n.inputs])
            outputs = OrderedSet([Flow[T](out) for out in n.outputs])

            # Create block
            b = cls(n.name, path, inputs, outputs)
            to_block[n] = b

            # Mark creation
            if n.creation is not None:
                b.creation = Flow[T](n.creation, creator=b)
                b.flavour = 'creator'
            
            # Add parent
            if n.parent:
                b.parent = to_block[n.parent]
                b.parent.children.append(b)
            assert b.parent is None or not b.parent.creation, "type T __init__ subcalls should be added to skip set"

        return to_block[root_node]


def get_lca_children_split[T](x: Block[T], y: Block[T]) -> tuple[Block[T], Block[T]]:
    """
    Find the last common ancestor of x and y.
    Then returns its two children a and b that are on paths to x and y respectively.
    """
    x_path = x.path_from_root
    y_path = y.path_from_root
    for i in range(min(len(x_path), len(y_path))):
        if x_path[i] != y_path[i]:
            return x_path[i], y_path[i]  # Found the first mismatch, return lca_child_to_x, lca_child_to_y
    raise ValueError("x and y are on the same path to root")


def update_ancestor_depths[T](b: Block[T]) -> None:
    """On return of a block b, set its depth to be after its inputs, update ancestor depths if necessary"""
    for inp in b.inputs:
        if inp.creator is None:
            continue
        b_ancestor, creator_ancestor = get_lca_children_split(b, inp.creator)
        if not 'constant' in creator_ancestor.tags:
            b.tags.discard('constant')  # node is downstream from inputs, not constant
        if b_ancestor.bot < creator_ancestor.top:  # current block must be above the parent block
            h_change = creator_ancestor.top - b_ancestor.bot
            b_ancestor.bot += h_change
            b_ancestor.top += h_change
    # TODO: can we update max shifting parent instead of fully looping over all inputs?
    # TODO: add copies if input creators are distant


def set_left_right[T](b: Block[T]) -> None:
    """Sets the left and right position of the block based on its parent"""
    if not b.parent:
        return
    current_block_width = max(b.levels) if b.levels else b.w
    if len(b.outputs) > current_block_width:
        current_block_width = len(b.outputs)
    horizontal_shift = b.parent.update_levels(b.bot, b.top, current_block_width)
    b.left += horizontal_shift
    b.right = b.left + current_block_width


def traverse[T](b: Block[T], order: Literal['call', 'return'] = 'call'
                   ) -> Generator[Block[T], None, None]:
    """Walks the call tree and yields each node."""
    if order == 'call':
        yield b
    for child in b.children:
        yield from traverse(child, order)
    if order == 'return':
        yield b


def get_missing_locations[T](
        init_b: Block[T], init_rel_depth: int, creator: Block[T]) -> set[Block[T]]:
    """Record copy_levels for blocks where creator creation is missing, leading to b missing it"""
    blocks_with_copies: set[Block[T]] = set()
    init_b_to_root = iter(init_b.path_from_root[::-1])
    descending_depths = reversed(range(creator.abs_y+creator.h, init_b.abs_y+init_rel_depth))
    b = next(init_b_to_root)
    for d in descending_depths:  # descend from init to creator

        # Move up the tree until we reach the next missing creation depth
        while b.abs_y > d:
            b = next(init_b_to_root)

        if d >= b.abs_y:
            # Record the missing creation for the current block and depth
            if d not in b.copy_levels:
                b.copy_levels[d] = []
            b.copy_levels[d].append(creator)
            blocks_with_copies.add(b)
            # TODO: order copy_levels in input order

    return blocks_with_copies


def get_blocks_with_missing_inputs[T](root: Block[T]) -> set[Block[T]]:
    """Find blocks with missing input instances"""
    available: dict[Block[T], list[OrderedSet[Flow[T]]]] = dict()  # b -> available instances at each depth
    required: dict[Block[T], list[OrderedSet[Flow[T]]]] = dict()  # b -> required instances at each depth
    blocks_with_copies: set[Block[T]] = set()

    for b in traverse(root.children[0]):
        available[b] = [OrderedSet() for _ in range(b.h + 1)]
        required[b] = [OrderedSet() for _ in range(b.h + 1)]
        available[b][0] |= b.inputs
        required[b][b.h] |= b.outputs
        for c in b.children:
            if c.creation:
                available[b][c.top].add(c.creation)  # since __init__ does not have outputs
            available[b][c.top] |= c.outputs
            required[b][c.bot] |= c.inputs

        # Check if all required are available
        for depth in range(b.h + 1):
            diff: OrderedSet[Flow[T]] = OrderedSet()
            available_data = {inst.data: inst for inst in available[b][depth]}
            diff = OrderedSet([inst for inst in required[b][depth] if inst.data not in available_data])
            if diff:
                for inst in diff:
                    assert inst.creator is not None
                    blocks_with_copies |= get_missing_locations(b, depth, inst.creator)
    
    return blocks_with_copies


def create_copy_blocks[T](blocks_with_copies: set[Block[T]]) -> None:
    """Create copy blocks for blocks that have missing input instances"""
    for b in blocks_with_copies:
        for i, creators in enumerate(b.copy_levels.values()):
            cwrap_name = f"copies-{i}" if len(b.copy_levels) > 1 else "copies"
            cwrap = Block[T](cwrap_name, b.path+f'.{cwrap_name}', parent=b)  # wrapper for individual copies
            for j, creator in enumerate(creators):
                flow = creator.creation
                assert flow is not None
                c_name = f"copy-{j}" if len(creators) > 1 else "copy"
                c = Block[T](c_name, cwrap.path+f'.{c_name}', parent=cwrap)
                c.inputs.add(flow)
                c.outputs.add(flow)
                c.tags.add('copy')
                c.flavour = 'copy'
                cwrap.children.append(c)
                cwrap.inputs.add(flow)
                cwrap.outputs.add(flow)
            cwrap.tags.add('copy')
            b.children.append(cwrap)
            # TODO: test with copying across several layers. Possibly this puts all copies at .abs_y level


def add_copies[T](root: Block[T]) -> None:
    blocks_with_copies = get_blocks_with_missing_inputs(root)
    create_copy_blocks(blocks_with_copies)


def create_input_blocks[T](root: Block[T]) -> list[Block[T]]:
    inp_blocks: list[Block[T]] = []
    for j, flow in enumerate(root.children[0].inputs):
        b = Block[T]('input', f'input[{j}]', creation=flow, tags={'input'}, flavour='input')
        inp_blocks.append(b)
    return inp_blocks


def set_flow_creator_for_io_of_each_block[T](root: Block[T], inp_blocks: list[Block[T]]) -> None:
    to_block: dict[T, Block[T]] = {}
    for inp_b in inp_blocks:
        assert inp_b.creation is not None
        inp_b.creation.creator = inp_b
        to_block[inp_b.creation.data] = inp_b
    for b in traverse(root, 'return'):
        if b.creation:
            to_block[b.creation.data] = b
        for flow in b.inputs | b.outputs:
            flow.creator = to_block[flow.data]


from typing import Any, Literal
from collections.abc import Callable, Generator
from circuits.utils.ftrace import FTracer
@dataclass
class Tracer[T]:
    tracked_type: type[T] | None = None
    skip: set[str] = field(default_factory=set[str])
    collapse: set[str] = field(default_factory=set[str])
    formatter: Callable[[T], str] = lambda x: str(x)

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Block[T]:
        assert self.tracked_type is not None
        ftracer = FTracer[T](self.tracked_type, self.skip, self.collapse)
        node = ftracer.run(func, *args, **kwargs)
        b = Block[T].from_root_node(node)
        self.set_layout(b)
        self.set_layout(b)
        self.set_layout(b)
        add_copies(b)
        self.set_layout(b)
        self.set_layout(b)
        self.set_formatting_info(b)
        b = b.unwrapped
        return b


    def set_layout(self, root: Block[T]) -> Block[T]:
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

        inp_blocks = create_input_blocks(root)
        set_flow_creator_for_io_of_each_block(root, inp_blocks)
        for b in traverse(root, order='return'):

            # Set creator/copy size to 1x1
            if b.flavour in {'creator', 'copy'}:
                b.top = b.bot + 1
                b.right = b.left + 1

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


    def mark_differences(self, root1: Block[T], root2: Block[T]) -> None:
        """Highlights the differences between two block trees"""
        for b1, b2 in zip(traverse(root1), traverse(root2)):
            assert b1.path == b2.path, f"Block paths do not match: {b1.path} != {b2.path}"
            if b1.out_str != b2.out_str:
                b1.tags.add('different')
                b2.tags.add('different')
                outs1 = [self.formatter(out.data) for out in b1.outputs]
                outs2 = [self.formatter(out.data) for out in b2.outputs]
                for out1, out2 in zip(outs1, outs2):
                    diff = ' ' if out1==out2 else out2
                    b1.outdiff += diff
                    b2.outdiff += diff


    def set_formatting_info(self, root: Block[T]) -> None:
        for b in traverse(root):
            b.nesting = b.parent.nesting + 1 if b.parent else -1

        for b in traverse(root, 'return'):
            b.max_leaf_nesting = max([c.max_leaf_nesting for c in b.children])+1 if b.children else 0
            b.out_str = "".join([self.formatter(out.data) for out in b.outputs])
