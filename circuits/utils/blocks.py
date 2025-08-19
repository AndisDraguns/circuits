from dataclasses import dataclass, field
from collections.abc import Callable, Generator
from typing import Literal, TypeVar

from circuits.utils.misc import OrderedSet
from circuits.utils.format import Bits
from circuits.utils.ftrace import CallNode, node_walk_generator

T = TypeVar('T')


@dataclass(eq=False)
class Block[T]:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    path: str
    depth: int  # Nesting depth in the call tree
    inputs: OrderedSet[T] = field(default_factory=OrderedSet[T])
    outputs: OrderedSet[T] = field(default_factory=OrderedSet[T])
    # is_creator: bool  # True if this node is __init__ of the tracked T instance
    created: T | None = None  # Instance created by this node, if any
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
    # tags: set[str] = field(default_factory=set[str])
    tags: set[str] = field(default_factory=lambda: {'constant'})

    # Output locations
    ox: int = 0  # x coordinate of the output
    oy: int = 0  # y coordinate of the output
    inp_indices: list[int] = field(default_factory=list[int])
    original: 'Block[T] | None' = None  # points to the original block (if this is a copy)
    copies: dict[int, 'Block[T]'] = field(default_factory=dict[int, 'Block[T]'])  # y -> copy


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
                path = f"{node_to_block[n.parent].path}." + f"{n.name}-{n.count}"
            b = Block[T](n.name, path, n.depth, inputs=n.inputs, outputs=n.outputs, created=n.created)
            node_to_block[n] = b
            if n.parent:
                b.parent = node_to_block[n.parent]
                node_to_block[n.parent].children.append(b)
        return node_to_block[root_node]

    def copy(self, ox: int) -> 'Block[T]':
        """Creates a copy of the block and its children."""
        original = self.original if self.original else self
        b = Block[T]('copy', self.path, self.depth, original=original, oy=self.oy+1, ox=ox)
        original.copies[self.oy+1] = b  # Add self to copies for the current height
        return b
    # def create_copy(self) -> 'Block[T]':
    #     """Creates a copy of the block and its children."""
    #     copy = Block[T](self.name, self.path, self.inputs, self.outputs, self.depth, self.created)
    #     copy.left = self.left
    #     copy.right = self.right
    #     copy.top = self.top
    #     copy.bot = self.bot
    #     copy.levels = self.levels[:]
    #     copy.tags = self.tags.copy()
    #     copy.children = [child.create_copy() for child in self.children]
    #     return copy

    # def highlight_differences(self, root2: 'Block[T]') -> None:
    #     """
    #     Highlights the differences between two call trees.
    #     Sets 'highlight' flag in for each block that differs from the corresponding block in the other tree.
    #     """
    #     gen1 = walk_generator(self)
    #     gen2 = walk_generator(root2)
    #     for val1, val2 in zip(gen1, gen2):
    #         node1, node2 = val1[0], val2[0]
    #         assert node1.path == node2.path, f"Node names do not match: {node1.path} != {node2.path}"
    #         if node1.out_str != node2.out_str:
    #             node1.tags.add('different')
    #             # node1.highlight = True
    #             for val1, val2 in zip(node1.outputs, node2.outputs):
    #                 val1_str = self.formatter(val1)
    #                 val2_str = self.formatter(val2)
    #                 node1.outdiff += ' ' if val1_str==val2_str else val2_str



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


def update_ancestor_heights(b: Block[T], instance_to_block: dict[T, Block[T]]) -> None:
    """
    On return of a block b, set blocks's height to be after its inputs, update ancestor heights if necessary.
    For each input, its creator block g is located. 
    """
    for inp in b.inputs:
        if inp not in instance_to_block:
            # b.tags.add('live')  # no trace -> input created outside of fn -> node is live (downstream from inputs)
            b.tags.discard('constant')
            continue
        g = instance_to_block[inp]
        b_ancestor, g_ancestor = get_lca_children_split(b, g)
        if not 'constant' in g_ancestor.tags:
            b.tags.discard('constant')
        if b_ancestor.bot < g_ancestor.top:  # current block must be above the parent block
            height_change = g_ancestor.top - b_ancestor.bot
            b_ancestor.bot += height_change
            b_ancestor.top += height_change

    # TODO: can we update max shifting parent instead of fully looping over all inputs?
    # TODO: add copies if input creators are distant


def process_creator_block(b: Block[T], instance_to_block: dict[T, Block[T]]) -> None:
    """Process the creator block"""
    assert b.created is not None, f"Expected creator block, got {b.name}"
    instance = b.created
    assert instance not in instance_to_block, f"b already in instance_to_block, b={b}"
    instance_to_block[instance] = b
    b.top += 1  # Set top height of b to 1, since it is the only leaf node
    b.right += 1  # Set the right position of b to 1, since it is the only leaf node


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


def get_levels(root: Block[T]) -> list[OrderedSet[T]]:
    """Gets the instances created in the tree sorted into levels based in their height"""
    levels: list[OrderedSet[T]] = [OrderedSet() for _ in range(root.top + 1)]  # Initialize levels list
    levels[0] = root.inputs
    for b, _ in walk_generator(root):
        if b.created is not None:
            levels[b.y+1].add(b.created)
    return levels

# from typing import NamedTuple
# @dataclass
# class Location[T]:
#     x: int
#     y: int
#     instance: T
#     block: Block[T]
#     copies: dict[int, 'Location[T]'] = field(default_factory=dict[int, 'Location[T]'])
#     def __post_init__(self):
#         self.copies[self.y] = self  # Add self to copies for the current height


def add_copies(root: Block[T]) -> list[list[Block[T]]]:
    """Gets the instances required in the tree sorted into levels based in their height"""
    levels = get_levels(root)

    instance_to_block: dict[T, Block[T]] = {}
    for b, _ in walk_generator(root):
        if b.created:
            instance_to_block[b.created] = b

    for i, level in enumerate(levels):
        for j, instance in enumerate(level):
            if instance not in instance_to_block:
                instance_to_block[instance] = Block[T]('input', f'input-{j}', -1, created=instance)
            b = instance_to_block[instance]
            b.ox = j  # Set x coordinate of the output
            b.oy = i  # Set y coordinate of the output
            b.copies[i] = b  # Add self to copies for the current height

    # for level in levels:
    #     # [instance_to_block[inst].path.split('.')[-4] for inst in level]
    #     print(len(level))
    # print("\n")

    # TODO: Idea:
    # get path from required to creator
    # make a leveled path lpath - i.e. each lpath el has y decreased by 1
    # that is - skip el that do not descend; and make several el when dropping within same block
    # copy tree? from creator to required blocks

    copies: list[list[Block[T]]] = [[] for _ in range(root.top + 1)]
    for b, _ in walk_generator(root):
        if b.name == 'gate':
        # if True:
            for inp_instance in b.inputs:
                inp = instance_to_block[inp_instance]
                assert inp.oy <= b.y, f"Instance required before it is created, {inp.oy}<={b.y}, inp:{inp.path}, b:{b.path}"
                prev_height = b.y
                inp_height = inp.oy
                # if inp_height < prev_height and b.name != 'gate':
                #     print(prev_height, inp_height, b.path, "    ", inp.path)
                #     continue
                # if b.name != 'gate':
                #     continue
                if inp_height == prev_height:
                    b.inp_indices.append(inp.ox)
                elif inp_height < prev_height:
                    if prev_height in inp.copies:
                        b.inp_indices.append(inp.copies[prev_height].ox)
                    else:
                        curr_height = inp.oy+1
                        while prev_height not in inp.copies:
                            if curr_height not in inp.copies:
                                last_copy = inp.copies[curr_height-1]
                                copy_block = last_copy.copy(ox=len(levels[curr_height])+1)
                                assert copy_block.original is not None and copy_block.original.created is not None
                                levels[curr_height].add(copy_block.original.created)
                                copies[curr_height].append(copy_block)
                            curr_height += 1
    return copies

# list[OrderedSet[Block[T]]]
def get_availability(root: Block[T]) -> None:
    instance_to_block: dict[T, Block[T]] = {}
    for b, _ in walk_generator(root):
        if b.created:
            instance_to_block[b.created] = b
    for j, inst in enumerate(root.inputs):
        if inst not in instance_to_block:
            instance_to_block[inst] = Block[T]('input', f'input-{j}', -1, created=inst)

    available: dict[Block[T], list[OrderedSet[T]]] = dict()  # b -> available instances at each height
    required: dict[Block[T], list[OrderedSet[T]]] = dict()  # b -> required instances at each height
    for b, _ in walk_generator(root, 'either'):
        # if 'constant' in b.tags:
        #     continue
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
            # diff = required[b][level].difference(available[b][level])
            diff = required[b][level]._d.keys() - available[b][level]._d.keys()
            if diff:
                for inst in diff:
                    b.tags.add('missing')
                    # instance_to_block[inst].tags.add('missing')
                print(f"n={len(diff)},\t level={level}: {b.path[-100:]} \t\t   {instance_to_block[list(diff)[0]].path[-100:]}")
                # print(f"Block {b.path} is missing {len(diff)} instances at level {level}: e.g. {instance_to_block[list(diff)[0]].path}")
                # try:
                #     z = f"Block {b.path} is missing {len(diff)} instances at level {level}: e.g. {instance_to_block[list(diff)[0]].path}"
                #     print(z)
                # except:
                #     print(f"! Block {b.path} is missing {len(diff)} instances at level {level})")
    print("Done")

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
        b = b.unwrapped
        get_availability(b)
        return b

    def postprocessing(self, root: Block[T]) -> Block[T]:
        """Processes the call tree"""
        instance_to_block: dict[T, Block[T]] = {}
        for b, _ in walk_generator(root, order='return'):
            # Set coordinates
            if b.created:
                process_creator_block(b, instance_to_block)
            update_ancestor_heights(b, instance_to_block)
            set_top(b)
            set_left_right(b)
            
            # Set tags and info
            b.max_leaf_depth = max([c.max_leaf_depth for c in b.children])+1 if b.children else 0
            b.out_str = "".join([self.formatter(out) for out in b.outputs])
            if any(['constant' not in c.tags for c in b.children]):
                b.tags.discard('constant')
            assert b.parent is None or not b.parent.created, "type T __init__ subcalls should be added to skip set"
        
        # Now that .left and .bot are finalized, set absolute coordinates
        for b, _ in walk_generator(root):
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
                outs1 = [self.formatter(out) for out in b1.outputs]
                outs2 = [self.formatter(out) for out in b2.outputs]
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
    tracer = Tracer[Bit]()
    msg1 = k.format("Reify semantics as referentless embeddings", clip=True)
    b1 = tracer.run(f, m=msg1, k=k)
    # print(b1)
    copies = add_copies(b1)
    for level in copies[2:]:
        path = [b.path.split('.')[-4] for b in level]
        print(len(level), path)
