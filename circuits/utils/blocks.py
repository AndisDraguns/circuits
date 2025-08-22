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
    indices: list[int] = field(default_factory=list[int])
    flat_index: int = 0  # Flattened index
    creator: 'Block[T] | None' = None
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

    # For visualising color and block output
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

    def info_str(self) -> str:
        """Returns a string representation of the node's info, excluding its children"""
        call_name = f"{self.name}"
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

    def __repr__(self):
        return f"{self.name}"

    def full_info(self) -> str:
        s = f"name: {self.name}\n"
        s += f"path: {self.path}\n"
        s += f"io: ({len(self.inputs)}→{len(self.outputs)})\n"
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
                if n.parent.counts[n.name] > 1:  # exclude count if function is only called once
                    path += f"-{n.count}"

            # Get input/output flows
            inputs = OrderedSet([Flow[T](inp, indices, i) for i, (inp, indices) in enumerate(n.inputs)])
            outputs = OrderedSet([Flow[T](out, indices, i) for i, (out, indices) in enumerate(n.outputs)])

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

        to_block[root_node].path = "root_wrapper"
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
    w = max(b.levels) if b.levels else b.w  # current_block_width
    if len(b.outputs) > w:
        w = len(b.outputs)
    index_shift = b.parent.update_levels(b.bot, b.top, w)
    b.left += index_shift
    b.right = b.left + w


def traverse[T](b: Block[T], order: Literal['call', 'return'] = 'call') -> Generator[Block[T], None, None]:
    """Walks the call tree and yields each node."""
    if order == 'call':
        yield b
    for child in b.children:
        yield from traverse(child, order)
    if order == 'return':
        yield b


def add_copies_to_block[T](b: Block[T]) -> None:
    required: dict[int, OrderedSet[Flow[T]]] = {d: OrderedSet() for d in range(b.h+1)}
    available: dict[int, OrderedSet[Flow[T]]] = {d: OrderedSet() for d in range(b.h+1)}
    required[b.h] = b.outputs
    available[0] = b.inputs
    for c in b.children:
        required[c.bot] |= c.inputs
        available[c.top] |= c.outputs
        if c.flavour == 'creator':
            available[c.top].add(c.creation)

    # descend from top to bot
    cwraps: list[Block[T]] = []
    for d in reversed(range(b.h + 1)):
        available_data = {inst.data: inst for inst in available[d]}
        copies = OrderedSet()
        for req in required[d]:
            if req.data not in available_data:

                if d==0:
                    raise ValueError(f"{req.creator.path} not available at {b.path} inputs")

                # create a copy
                copy = Block[T]("c", "c", flavour='copy')
                copy_outflow = Flow[T](req.data, creator=req.creator)
                copy_inflow = Flow[T](req.data, creator=req.creator)
                copy.outputs.add(copy_outflow)
                copy.inputs.add(copy_inflow)
                available_data.update({req.data: copy_outflow})
                required[d-1].add(copy_inflow)
                copies.add(copy)

            req.prev = available_data[req.data]

        # create a copies wrapper for better visualisation
        if copies:
            cwrap = Block[T]("cw", "cw", parent=b)
            for i, copy in enumerate(copies):
                cwrap.children.append(copy)
                copy.parent = cwrap
                cwrap.inputs |= copy.inputs
                cwrap.outputs |= copy.outputs
            b.children.append(cwrap)
            cwraps.append(cwrap)

    # set copies formatting info
    for i, cwrap in enumerate(reversed(cwraps)):
        cwrap.name = f"copies-{i}" if len(cwraps) > 1 else "copies"
        cwrap.path = b.path+f".{cwrap.name}"
        cwrap.tags.add('copy')
        for j, copy in enumerate(cwrap.children):
            copy.name = f"copy-{j}" if len(copies) > 1 else "copy"
            copy.path = cwrap.path+f".{copy.name}"
            copy.tags.add('copy')


def add_copies[T](root: Block[T]) -> None:
    for b in traverse(root, 'return'):
        add_copies_to_block(b)


def create_input_blocks[T](root: Block[T]) -> list[Block[T]]:
    inp_blocks: list[Block[T]] = []
    for j, flow in enumerate(root.children[0].inputs):
        b = Block[T]('input', f'input[{j}]', creation=flow, tags={'input'}, flavour='input')
        inp_blocks.append(b)
    return inp_blocks


def set_flow_creator_for_io_of_each_block[T](root: Block[T], inp_blocks: list[Block[T]]) -> None:
    to_block: dict[T, Block[T]] = {}
    for inp_b in inp_blocks:
        flow = inp_b.creation
        assert flow is not None
        flow.creator = inp_b
        to_block[flow.data] = inp_b
    for b in traverse(root, 'return'):
        if b.creation:
            to_block[b.creation.data] = b
        for flow in b.inputs | b.outputs:
            flow.creator = to_block[flow.data]


from typing import Any
from circuits.utils.ftrace import FTracer
@dataclass
class Tracer[T]:
    tracked_type: type[T] | None = None
    collapse: set[str] = field(default_factory=set[str])
    skip: set[str] = field(default_factory=set[str])
    formatter: Callable[[T], str] = lambda x: str(x)

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Block[T]:
        assert self.tracked_type is not None
        ftracer = FTracer[T](self.tracked_type, self.collapse, self.skip)
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
