from dataclasses import dataclass, field
from collections.abc import Generator
from typing import Literal

from circuits.neurons.core import Signal
from circuits.utils.misc import OrderedSet
from circuits.utils.format import Bits
from circuits.utils.ftrace import CallNode, node_walk_generator

@dataclass(eq=False)
class Block:
    """Represents a function call with its Signal inputs/outputs"""
    name: str
    path: str
    inputs: OrderedSet[Signal]
    outputs: OrderedSet[Signal]
    depth: int  # Nesting depth in the call tree
    parent: 'Block | None' = None
    children: list['Block'] = field(default_factory=list['Block'])

    bot: int = 0  # Bottom height of the node in the call tree (relative to parent.top)
    top: int = 0  # Top height of the node in the call tree (relative to self.bot)
    left: int = 0  # left position of the node in the call tree (relative to parent.left)
    right: int = 0  # right position of the node in the call tree (relative to self.left)
    levels: list[int] = field(default_factory=list[int])  # level widths of the node in the call tree
    x: int = 0  # Absolute x coordinate (leftmost edge)
    y: int = 0  # Absolute y coordinate (bottom edge)
    max_leaf_depth: int = -1

    out_str: str = ""  # String representation of the outputs
    outdiff: str = ""  # String representation of the outputs that differ from some other node

    highlight: bool = False  # Whether to highlight this node
    is_live: bool = False  # live = generates signals that are downstream from inputs

    tags: set[str] = field(default_factory=set[str])


    @property
    def fpath(self) -> tuple['Block', ...]:
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
        assert self.top - self.bot >= 0, f"self.top - self.bot = {self.top - self.bot}, {self.name}"
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
        # return ""
        indent = "  " * level
        info = self.info_str()
        child_names = "".join(f"\n{c.__str__(level + 1, hide)}" for c in self.children if c.name not in hide)
        res = f"{indent}{info}{child_names}"
        return res
    
    def __repr__(self):
        return f"b {self.name}"

    def full_info(self) -> str:
        s = f"name-count: {self.name}\n"
        s += f"io: ({len(self.inputs)}→{len(self.outputs)})\n"
        s += f"full_name: {self.path}\n"
        s += f"depth of nesting: {self.depth}\n"
        s += f"x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}\n"
        s += f"is_live: {self.is_live}, highlight: {self.highlight}\n"
        s += f"out_str: '{self.out_str}'\n"
        if self.outdiff:
            s += f"outdiff: '{self.outdiff}'\n"
        return s
    

    @classmethod
    def from_node(cls, root_node: CallNode) -> 'Block':
        node_to_block: dict[CallNode, Block] = {}
        for n, _ in node_walk_generator(root_node, order='call'):
            path = f"{n.name}-{n.count}"
            if n.parent is not None:
                path = f"{node_to_block[n.parent].path}." + path
                # print("path:",path)
            b = Block(n.name, path, n.inputs, n.outputs, n.depth)
            node_to_block[n] = b
            if n.parent:
                b.parent = node_to_block[n.parent]
                node_to_block[n.parent].children.append(b)
        return node_to_block[root_node]


    def process(self) -> 'Block':
        post_process_trace(self)
        return self


    def highlight_differences(self, root2: 'Block') -> None:
        """
        Highlights the differences between two call trees.
        Sets 'highlight' flag in for each call node that differs from the corresponding node in the other tree.
        """
        gen1 = walk_generator(self)
        gen2 = walk_generator(root2)
        for val1, val2 in zip(gen1, gen2):
            node1, node2 = val1[0], val2[0]
            assert node1.path == node2.path, f"Node names do not match: {node1.path} != {node2.path}"
            if node1.out_str != node2.out_str:
                # print(node1.path)
                node1.highlight = True
                node1.outdiff = "".join([' ' if s1==s2 else s2 for s1, s2 in zip(node1.out_str, node2.out_str)])
            # else:
            #     if node2.name != 'root_wrapper_fn':
            #         print(node2.name)
            #         assert False, f"{node2.name}"

# def blocks_from_nodes(root_node: CallNode) -> Block:
#     node_to_block: dict[CallNode, Block] = {}
#     for n, _ in node_walk_generator(root_node, order='call'):
#         path = f"{n.name}-{n.count}"
#         if n.parent is not None:
#             path = f"{n.parent.full_name}." + path
#         b = Block(n.name, path, n.inputs, n.outputs, n.depth)
#         node_to_block[n] = b
#         if n.parent:
#             b.parent = node_to_block[n.parent]
#             node_to_block[n.parent].children.append(b)
#     return node_to_block[root_node]


def get_lca_children_split(x: Block, y: Block) -> tuple[Block, Block]:
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


def update_ancestor_heights(n: Block) -> None:
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


def process_gate_return(g: Block) -> None:
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


def set_top(node: Block) -> None:
    """Sets the top height of the node based on its children"""
    if not node.children:
        return
    block_height = max([c.top for c in node.children])
    if node.name == 'gate':
        block_height = 1
    node.top = node.bot + block_height


def set_left_right(node: Block) -> None:
    """Sets the left and right position of the node based on its parent"""
    if not node.parent:
        return
    current_block_width = max(node.levels) if node.levels else node.right - node.left
    if len(node.outputs) > current_block_width:
        current_block_width = len(node.outputs)
    horizontal_shift = node.parent.add(node.bot, node.top, current_block_width)
    node.left += horizontal_shift
    node.right = node.left + current_block_width


def walk_generator(node: Block, order: Literal['call', 'return', 'both', 'either'] = 'either'
                   ) -> Generator[tuple[Block, Literal['call', 'return']], None, None]:
    """Walks the call tree and yields each node."""
    if order in {'call', 'both', 'either'}:
        yield node, 'call'
    for child in node.children:
        yield from walk_generator(child, order)
    if order in {'return', 'both'}:
        yield node, 'return'


def post_process_trace(root: Block) -> Block:
    """Processes the call tree"""
    for b, _ in walk_generator(root, order='return'):
        b.max_leaf_depth = max([c.max_leaf_depth for c in b.children])+1 if b.children else 0

        b.out_str = Bits(list(b.outputs)).bitstr

        # The following must be on return
        # Sets node's coordinates
        if b.name == 'gate':
            process_gate_return(b)
        update_ancestor_heights(b)
        set_top(b)
        set_left_right(b)
        if any([c.is_live for c in b.children]):
            b.is_live = True

    
    # Now that .left and .bot are finalized, we can  set absolute coordinates
    for b, _ in walk_generator(root):
        if b.parent is not None:
            b.x = b.left + b.parent.x
            b.y = b.bot + b.parent.y

    return root





# def highlight_differences(root1: Block, root2: Block) -> None:
#     """
#     Highlights the differences between two call trees.
#     Sets 'highlight' flag in for each call node that differs from the corresponding node in the other tree.
#     """
#     gen1 = walk_generator(root1)
#     gen2 = walk_generator(root2)
#     for val1, val2 in zip(gen1, gen2):
#         node1, node2 = val1[0], val2[0]
#         assert node1.path == node2.path, f"Node names do not match: {node1.path} != {node2.path}"
#         if node1.out_str != node2.out_str:
#             node2.highlight = True
#             node2.outdiff = "".join([' ' if s1==s2 else s1 for s1, s2 in zip(node1.out_str, node2.out_str)])


# if __name__ == '__main__':
    
#     from circuits.utils.ftrace import Tracer
#     tracer = Tracer(use_defaults=True)

#     from circuits.examples.keccak import Keccak
#     from circuits.neurons.core import Bit
#     from circuits.utils.format import Bits
#     def test(message: Bits, k: Keccak) -> list[Bit]:
#         hashed = k.digest(message)
#         return hashed.bitlist
#     k = Keccak(c=10, l=0, n=1, pad_char='_')
#     phrase = "Reify semantics as referentless embeddings"
#     message = k.format(phrase, clip=True)
#     trace = tracer.run(test, message=message, k=k)
#     hide = {'gate'}
#     # print(trace.root.__str__(hide=hide))

#     b = blocks_from_nodes(trace.root)
#     _, max_depth = post_process_trace(b)
#     # assert False
#     # print("a")
    
#     # print(len(b.children))
#     print(b.__str__(hide=hide))