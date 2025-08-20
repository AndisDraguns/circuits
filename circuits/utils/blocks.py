from dataclasses import dataclass, field
from collections.abc import Callable, Generator
from typing import Literal, TypeVar

from circuits.utils.misc import OrderedSet
from circuits.utils.format import Bits
from circuits.utils.ftrace import CallNode, node_walk_generator

T = TypeVar('T')


@dataclass(eq=False)
class Flow[T]:
    """Represents data flow between blocks"""
    # Same data instance can occur at many positions, e.g. due to copying
    data: T
    height: int = -1  # Absolute vertical position
    index: int = -1  # Absolute horizontal position
    creator: 'Block[T] | None' = None
    prev_copy: 'Block[T] | None' = None
    prev: 'Flow[T] | None' = None  # Previous flow on the same height


@dataclass(eq=False)
class Block[T]:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    path: str
    depth: int  # Nesting depth in the call tree
    inputs: OrderedSet[Flow[T]] = field(default_factory=OrderedSet[Flow[T]])
    outputs: OrderedSet[Flow[T]] = field(default_factory=OrderedSet[Flow[T]])
    created: Flow[T] | None = None  # Instance created by this node, if any
    parent: 'Block[T] | None' = None
    children: list['Block[T]'] = field(default_factory=list['Block[T]'])

    # Positioning
    bot: int = 0  # Bottom height of the node in the call tree (relative to parent.top)
    top: int = 0  # Top height of the node in the call tree (relative to self.bot)
    left: int = 0  # left position of the node in the call tree (relative to parent.left)
    right: int = 0  # right position of the node in the call tree (relative to self.left)
    levels: list[int] = field(default_factory=list[int])  # level widths of the node in the call tree
    x: int = 0  # Absolute x coordinate (leftmost edge)
    y: int = 0  # Absolute y coordinate (bottom edge)
    max_leaf_depth: int = -1

    # For visualizng color and block output
    formatter: Callable[[T], str] = lambda x: str(x)  # Function to format tracked instances
    out_str: str = ""  # String representation of the outputs
    outdiff: str = ""  # String representation of the outputs that differ from some other node
    tags: set[str] = field(default_factory=lambda: {'constant'})

    # Output locations
    ox: int = 0  # x coordinate of the output
    oy: int = 0  # y coordinate of the output
    inp_indices: list[int] = field(default_factory=list[int])
    original: 'Block[T] | None' = None  # points to the original block (if this is a copy)
    copies: dict[int, 'Block[T]'] = field(default_factory=dict[int, 'Block[T]'])  # y -> copy

    copy_levels: dict[int, list['Block[T]']] = field(default_factory=dict[int, list['Block[T]']])  # level -> blocks to copy


    @property
    def fpath(self) -> tuple['Block[T]', ...]:
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
        """Adds a child node at bot-top height. width = child width.
        Updates self.levels widths. Returns the new child left position"""
        if len(self.levels) < top:
            self.levels.extend(0 for _ in range(len(self.levels), top))
        heights = list(range(bot, top))
        widths = [self.levels[h] for h in heights]
        new_left = max(widths) if widths else 0  # Find the maximum width at the heights
        self.levels[bot:top] = [new_left + width] * len(heights)  # Update all levels in the range to child right
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
        s = f"name-count: {self.name}\n"
        s += f"io: ({len(self.inputs)}→{len(self.outputs)})\n"
        s += f"path: {self.path}\n"
        s += f"depth of nesting: {self.depth}\n"
        s += f"x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}\n"
        s += f"tags: {self.tags}\n"
        s += f"out_str: '{self.out_str}'\n"
        if self.outdiff:
            s += f"outdiff: '{self.outdiff}'\n"
        return s

    @classmethod
    def from_node(cls, root_node: CallNode[T]) -> 'Block[T]':
        node_to_block: dict[CallNode[T], Block[T]] = {}
        for n, _ in node_walk_generator(root_node, order='call'):
            path = ""
            if n.parent is not None:
                path = f"{node_to_block[n.parent].path}"
                if n.parent.parent is not None:
                    path += "."
                path += f"{n.name}" 
                if n.parent.fn_counts[n.name] > 1:  # exclude count if function is only called once
                    path += f"-{n.count}"
            inputs = OrderedSet([Flow[T](inp) for inp in n.inputs])
            outputs = OrderedSet([Flow[T](out) for out in n.outputs])
            created = None
            if n.created is not None:
                created = Flow[T](n.created)
            b = cls(n.name, path, n.depth, inputs, outputs, created=created)
            node_to_block[n] = b
            if n.parent:
                b.parent = node_to_block[n.parent]
                b.parent.children.append(b)
        return node_to_block[root_node]

    # def copy(self, ox: int) -> 'Block[T]':
    #     """Creates a copy of the block"""
    #     original = self.original if self.original else self
    #     b = Block[T]('copy', self.path, self.depth, original=original, oy=self.oy+1, ox=ox)
    #     original.copies[self.oy+1] = b  # Add self to copies for the current height
    #     return b


def lowest_common_ancestor(x: Block[T], y: Block[T]) -> Block[T]:
    x_path = x.fpath
    y_path = y.fpath
    for i in range(min(len(x_path), len(y_path))):
        if x_path[i] != y_path[i]:
            return x_path[i-1]  # Found the first mismatch
    # x and y are on the same path to root:
    if len(x_path) < len(y_path):
        return x_path[-1]
    else:
        return y_path[-1]


def get_path_between_blocks(x: Block[T], y: Block[T]) -> tuple[Block[T], ...]:
    """Gets path from x to y"""
    lca = lowest_common_ancestor(x, y)
    lca_to_x = x.fpath[x.fpath.index(lca):]
    lca_to_y = y.fpath[y.fpath.index(lca):]
    x_to_lca = lca_to_x[::-1]
    x_to_y = x_to_lca[:-1] + lca_to_y
    return tuple(x_to_y)


def get_lca_children_split(x: Block[T], y: Block[T]) -> tuple[Block[T], Block[T]]:
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


def update_ancestor_heights(b: Block[T]) -> None:
    """On return of a block b, set its height to be after its inputs, update ancestor heights if necessary"""
    for inp in b.inputs:
        if inp.creator is None:
            # input created outside of the root fn -> node is downstream from inputs, not constant
            b.tags.discard('constant')
            continue
        assert inp.creator is not None
        b_ancestor, creator_ancestor = get_lca_children_split(b, inp.creator)
        if not 'constant' in creator_ancestor.tags:
            b.tags.discard('constant')
        if b_ancestor.bot < creator_ancestor.top:  # current block must be above the parent block
            height_change = creator_ancestor.top - b_ancestor.bot
            b_ancestor.bot += height_change
            b_ancestor.top += height_change
    # TODO: can we update max shifting parent instead of fully looping over all inputs?
    # TODO: add copies if input creators are distant


def process_creator_block(b: Block[T]) -> None:
    """Process the creator block"""
    assert b.created is not None, f"Expected creator block, got {b.name}"
    # instance = b.created
    # assert instance not in instance_to_block, f"b already in instance_to_block, b={b}"
    if b.created.creator is None:
        b.created.creator = b
    # if instance not in instance_to_block:
        # instance_to_block[instance] = b
    else:
        # print(b.path, b.created.creator.path)
        if not 'copy' in b.tags:
            assert b.path == b.created.creator.path
        # assert 'copy' in b.tags
    b.top = b.bot + 1  # Set top height of b to 1, since it is the only leaf node
    b.right = b.left + 1  # Set the right position of b to 1, since it is the only leaf node


def set_top(node: Block[T]) -> None:
    """Sets the top height of the node based on its children"""
    if not node.children:
        return
    block_height = max([c.top for c in node.children])
    if node.created:
        block_height = 1
    node.top = node.bot + block_height


def set_left_right(b: Block[T]) -> None:
    """Sets the left and right position of the block based on its parent"""
    if not b.parent:
        return
    current_block_width = max(b.levels) if b.levels else b.w
    if len(b.outputs) > current_block_width:
        current_block_width = len(b.outputs)
    horizontal_shift = b.parent.update_levels(b.bot, b.top, current_block_width)
    b.left += horizontal_shift
    b.right = b.left + current_block_width


def walk_generator(node: Block[T], order: Literal['call', 'return', 'both', 'either'] = 'either'
                   ) -> Generator[tuple[Block[T], Literal['call', 'return']], None, None]:
    """Walks the call tree and yields each node."""
    if order in {'call', 'both', 'either'}:
        yield node, 'call'
    for child in node.children:
        yield from walk_generator(child, order)
    if order in {'return', 'both'}:
        yield node, 'return'


# def get_levels(root: Block[T]) -> list[OrderedSet[T]]:
#     """Gets the instances created in the tree sorted into levels based in their height"""
#     levels: list[OrderedSet[T]] = [OrderedSet() for _ in range(root.top + 1)]  # Initialize levels list
#     levels[0] = root.inputs
#     for b, _ in walk_generator(root):
#         if b.created is not None:
#             levels[b.y+1].add(b.created)
#     return levels


def get_missing_locations(missing_b: Block[T], level: int, creator: Block[T]) -> set[Block[T]]:
    missing_to_root = iter(missing_b.fpath[::-1])
    curr_height = missing_b.y + level
    creator_height = creator.y + creator.h
    curr_block = next(missing_to_root)
    blocks_with_copies: set[Block[T]] = set()
    for height in reversed(range(creator_height, curr_height)):
        while curr_block.y > height:
            curr_block = next(missing_to_root)
        if height >= curr_block.y:
            if height not in curr_block.copy_levels:
                curr_block.copy_levels[height] = []
            curr_block.copy_levels[height].append(creator)
            blocks_with_copies.add(curr_block)
            # TODO: order copy_levels in input order
    return blocks_with_copies


def get_missing_inputs(root: Block[T]) -> None:
    # Create instance to block mapping, including disconnected blocks for root inputs
    # instance_to_block: dict[T, Block[T]] = {}
    # for b, _ in walk_generator(root.children[0]):
    #     if b.created:
    #         instance_to_block[b.created] = b
    for j, flow in enumerate(root.children[0].inputs):
        if flow.creator is None:
            flow.height = -1
            flow.index = j
            flow.creator = Block[T]('input', f'input[{j}]', -1, created=flow)
        # if inst not in instance_to_block:
        #     instance_to_block[inst] = Block[T]('input', f'input-{j}', -1, created=inst)

    # Find missing instances
    available: dict[Block[T], list[OrderedSet[Flow[T]]]] = dict()  # b -> available instances at each height
    required: dict[Block[T], list[OrderedSet[Flow[T]]]] = dict()  # b -> required instances at each height
    missing: dict[Block[T], dict[int, OrderedSet[Block[T]]]] = dict()  # b -> missing instances at each height
    blocks_with_copies: set[Block[T]] = set()
    for b, _ in walk_generator(root.children[0], 'either'):
        available[b] = [OrderedSet() for _ in range(b.h + 1)]
        required[b] = [OrderedSet() for _ in range(b.h + 1)]
        available[b][0] |= b.inputs
        required[b][b.h] |= b.outputs
        for c in b.children:
            if b.name == 'gate' and c.name == '__init__':
                assert c.created
                available[b][c.top].add(c.created)  # since __init__ does not have outputs
            available[b][c.top] |= c.outputs
            required[b][c.bot] |= c.inputs
        # Check if all required are available
        for level in range(b.h + 1):
            # diff = required[b][level] - available[b][level]
            diff: OrderedSet[Flow[T]] = OrderedSet()
            available_data = {inst.data: inst for inst in available[b][level]}
            for j, req_flow in enumerate(required[b][level]):
                # req_flow.height = b.y + level
                # req_flow.index = j
                if req_flow.data in available_data:
                    req_flow.prev = available_data[req_flow.data]
                else:
                    diff.add(req_flow)
                    # b.tags.add('missing')
            # required_data = OrderedSet([inst.data for inst in required[b][level]])
            diff = OrderedSet([inst for inst in required[b][level] if inst.data not in available_data])
            if diff:
                for inst in diff:
                    b.tags.add('missing')
                    assert inst.creator is not None, f"{inst}"
                    blocks_with_copies |= get_missing_locations(b, level, inst.creator)
                # print(f"n={len(diff)},\t level={level}: {b.path[-100:]} \t   {instance_to_block[list(diff)[0]].path[-100:]}")
                if b not in missing:
                    missing[b] = {}
                missing[b][level] = OrderedSet([inst.creator for inst in diff if inst.creator is not None])

    # Create copy blocks for missing instances
    for b in blocks_with_copies:
        for i, (_, copies) in enumerate(b.copy_levels.items()):
            name = "copies"
            if len(b.copy_levels)>1:
                name += f"-{i}"
            copies_block = Block[T](name, b.path+f'.{name}', b.depth+1, parent=b)
            for j, c in enumerate(copies):
                assert c.created is not None
                c_name = "copy"
                if len(copies) > 1:
                    c_name += f"-{j}"
                new_block = Block[T](c_name, copies_block.path+f'.{c_name}', copies_block.depth+1, parent=copies_block,
                                     inputs=OrderedSet([c.created]),
                                     outputs=OrderedSet([c.created]))
                new_block.tags.add('copy')
                copies_block.children.append(new_block)
                copies_block.inputs.add(c.created)
                copies_block.outputs.add(c.created)
            copies_block.tags.add('copy')
            b.children.append(copies_block)
            # TODO: test with copying across several layers
            # Possibly this puts all copies at .y level


from typing import Any, Literal
from collections.abc import Callable, Generator
from circuits.utils.ftrace import FTracer
@dataclass
class Tracer[T]:
    skip: set[str] = field(default_factory=set[str])
    collapse: set[str] = field(default_factory=set[str])
    formatter: Callable[[T], str] = lambda x: str(x)

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Block[T]:
        from circuits.neurons.core import Signal
        ftracer = FTracer[T](Signal, self.skip, self.collapse)
        node = ftracer.run(func, *args, **kwargs)
        b = Block[T].from_node(node)
        self.postprocessing(b)
        self.postprocessing(b)
        self.postprocessing(b)
        get_missing_inputs(b)
        self.postprocessing(b)
        b = b.unwrapped
        # get_missing_inputs(b)
        return b

    def postprocessing(self, root: Block[T]) -> Block[T]:
        """Processes the call tree"""
        # Reset if postprocessing was already called
        for b, _ in walk_generator(root):
            b.bot = 0
            b.top = 0
            b.left = 0
            b.right = 0
            b.levels = []
            b.x = 0
            b.y = 0
            b.max_leaf_depth = -1
            # b.tags.discard('missing')


    # levels: list[int] = field(default_factory=list[int])  # level widths of the node in the call tree
    # x: int = 0  # Absolute x coordinate (leftmost edge)
    # y: int = 0  # Absolute y coordinate (bottom edge)
    # max_leaf_depth: int = -1

        instance_to_block: dict[T, Block[T]] = {}
        for j, flow in enumerate(root.children[0].inputs):
            flow.height = -1
            flow.index = j
            flow.creator = Block[T]('input', f'input[{j}]', -1, created=flow)
            instance_to_block[flow.data] = flow.creator

        for b, _ in walk_generator(root, order='return'):
            # Set coordinates
            if b.created:
                process_creator_block(b)
                instance_to_block[b.created.data] = b
            for flow in b.inputs | b.outputs:
                flow.creator = instance_to_block[flow.data]
            if 'copy' in b.tags:
                b.top = b.bot + 1
                b.right = b.left + 1
            update_ancestor_heights(b)
            set_top(b)
            set_left_right(b)
            
            # Set tags and info
            b.max_leaf_depth = max([c.max_leaf_depth for c in b.children])+1 if b.children else 0
            b.out_str = "".join([self.formatter(out.data) for out in b.outputs])
            if any(['constant' not in c.tags for c in b.children]):
                b.tags.discard('constant')
            assert b.parent is None or not b.parent.created, "type T __init__ subcalls should be added to skip set"
        
        # Now that .left and .bot are finalized, set absolute coordinates
        for b, _ in walk_generator(root):
            # for flow in b.inputs | b.outputs:
            #     flow.creator = instance_to_block[flow.data]
            if b.parent is not None:
                b.x = b.left + b.parent.x
                b.y = b.bot + b.parent.y

        return root


    def mark_differences(self, root1: Block[T], root2: Block[T]) -> None:
        """Highlights the differences between two block trees"""
        for v1, v2 in zip(walk_generator(root1), walk_generator(root2)):
            b1, b2 = v1[0], v2[0]
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


# Example usage
if __name__ == '__main__':
    from circuits.neurons.core import Bit
    from circuits.examples.keccak import Keccak
    def f(m: Bits, k: Keccak) -> list[Bit]:
        return k.digest(m).bitlist
    k = Keccak(c=10, l=0, n=1, pad_char='_')
    tracer = Tracer[Bit](collapse = {'__init__', 'outgoing', 'step'})
    msg1 = k.format("Reify semantics as referentless embeddings", clip=True)
    b1 = tracer.run(f, m=msg1, k=k)
    # print(b1)
    # copies = add_copies(b1)
    # for level in copies[2:]:
    #     path = [b.path.split('.')[-4] for b in level]
    #     print(len(level), path)
