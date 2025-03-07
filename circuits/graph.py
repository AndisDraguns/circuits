from dataclasses import dataclass, field
from collections import defaultdict

from circuits.core import Signal


@dataclass(eq=False)
class Node:
    val: int | float | bool = -1  # stores Signal activation, used for debugging
    parents: set["Node"] = field(default_factory=set)
    children: set["Node"] = field(default_factory=set)
    weights: dict["Node", int | float] = field(default_factory=lambda: {})
    bias: int | float = 0
    depth: int | None = None
    column: int | None = None

    __hash__ = object.__hash__  # hash(id)

    def add_parent(self, parent: "Node", weight: int | float = 0):
        self.parents.add(parent)
        parent.children.add(self)
        self.weights[parent] = weight

    def replace_parent(self, old_parent: "Node", new_parent: "Node"):
        self.add_parent(new_parent, self.weights[old_parent])
        self.parents.remove(old_parent)
        old_parent.children.remove(self)
        del self.weights[old_parent]


@dataclass
class Graph:
    """Neural network graph"""

    layers: list[list[Node]]

    def __init__(self, inputs: list[Signal], outputs: list[Signal]) -> None:
        inp_nodes, out_nodes = self.load_nodes(inputs, outputs)
        self.layers = self.build_layers(inp_nodes, out_nodes)

    @classmethod
    def build_layers(cls, inputs: list[Node], outputs: list[Node]) -> list[list[Node]]:
        layers = cls.initialize_layers(inputs)
        layers = cls.set_output_layer(layers, outputs)
        layers = cls.ensure_adjacent_parents(layers)
        return layers

    @staticmethod
    def fuse_constants_into_biases(constants: set[Node], excluded: set[Node]) -> None:
        # Fold constants into thresholds
        while constants:
            new_constants: set["Node"] = set()
            for c in constants:
                value = c.bias + 1
                for child in c.children:
                    w = child.weights[c]
                    child.bias += value * w
                    del child.weights[c]
                    child.parents.remove(c)
                    if len(child.parents) == 0 and child not in excluded:
                        new_constants.add(child)  # treat any new leaf nodes
            constants = new_constants

    @classmethod
    def load_nodes(
        cls, inp_signals: list[Signal], out_signals: list[Signal]
    ) -> tuple[list[Node], list[Node]]:
        """Create nodes from signals"""
        inp_nodes = [Node(int(s.activation)) for s in inp_signals]
        out_nodes = [Node(int(s.activation)) for s in out_signals]
        inp_set = set(inp_nodes)
        out_set = set(out_nodes)
        nodes = {k: v for k, v in zip(inp_signals + out_signals, inp_nodes + out_nodes)}
        signals = {v: k for k, v in nodes.items()}
        seen: set[Node] = set()
        frontier = out_nodes
        disconnected = True
        constants: set[Node] = set()

        # Go backwards from output nodes to record all connections
        while frontier:
            new_frontier: set["Node"] = set()
            seen.update(frontier)
            for child in frontier:

                # Stop at inputs, they could have parents
                if child in inp_set:
                    disconnected = False
                    continue

                # Record parents of frontier nodes
                neuron = signals[child].source
                child.bias = neuron.bias
                for i, p in enumerate(neuron.incoming):
                    if p not in nodes:
                        nodes[p] = Node(int(p.activation))
                        signals[nodes[p]] = p
                    parent = nodes[p]
                    if parent not in seen:
                        new_frontier.add(parent)
                    child.add_parent(parent, weight=neuron.weights[i])

                if len(child.parents) == 0:
                    constants.add(child)

            frontier = new_frontier

        cls.fuse_constants_into_biases(constants, inp_set | out_set)

        assert not disconnected, "Outputs not connected to inputs"
        return inp_nodes, out_nodes

    @staticmethod
    def initialize_layers(inp_nodes: list[Node]) -> list[list[Node]]:
        """Places signals into layers. Sets depth as distance from input nodes"""
        n_parents_computed: defaultdict[Node, int] = defaultdict(
            int
        )  # default nr parents computed = 0
        layers = [inp_nodes]
        frontier = set(inp_nodes)
        for inp in frontier:
            inp.depth = 0
        depth = 0
        while frontier:
            new_frontier: set[Node] = set()
            for parent in frontier:
                for child in parent.children:
                    n_parents_computed[child] += 1
                    if n_parents_computed[child] == len(
                        child.parents
                    ):  # parents computed
                        new_frontier.add(child)
                        child.depth = depth + 1  # child is in the next layer
            frontier = new_frontier
            layers.append(list(frontier))
            depth += 1
        return layers

    @staticmethod
    def set_output_layer(
        layers: list[list[Node]], out_nodes: list[Node]
    ) -> list[list[Node]]:
        """Ensure that all output nodes are on the last layer"""
        out_set = set(out_nodes)
        out_depths = {node.depth for node in out_set if node.depth is not None}
        for depth in out_depths:  # delete output nodes
            layers[depth] = [node for node in layers[depth] if node not in out_set]
        max_depth = len(layers) - 1
        layers[max_depth] = out_nodes[:]
        for out in out_set:
            out.depth = max_depth
        return layers

    @staticmethod
    def ensure_adjacent_parents(layers: list[list[Node]]) -> list[list[Node]]:
        """Copy signals to next layers, ensuring child.depth==parent.depth+1"""
        copies_by_layer: list[list[Node]] = [[] for _ in range(len(layers))]
        for layer_idx, layer in enumerate(layers):
            for node in layer:

                # Stop at outputs
                if len(node.children) == 0:
                    continue

                max_child_depth = max(
                    [c.depth for c in node.children if c.depth is not None]
                )
                n_missing_layers = max_child_depth - (layer_idx + 1)
                if n_missing_layers <= 0:
                    continue

                # Create chain of copies
                copy_chain: list[Node] = []
                prev = node
                for depth in range(layer_idx + 1, layer_idx + n_missing_layers + 1):
                    curr = Node(prev.val)
                    curr.depth = depth
                    curr.bias = -1
                    curr.add_parent(prev, weight=1)
                    copy_chain.append(curr)
                    prev = curr

                # Redirect children to appropriate copies
                for child in list(node.children):
                    if child.depth is None:
                        raise ValueError("Child depth must be set")
                    elif child.depth <= layer_idx + 1:
                        continue
                    new_parent = copy_chain[child.depth - layer_idx - 2]
                    child.replace_parent(node, new_parent)

                # Add copies to their respective layers
                for i, copy_node in enumerate(copy_chain):
                    copies_by_layer[layer_idx + 1 + i].append(copy_node)

        # Add copies and record indices
        for i, layer in enumerate(layers):
            layer.extend(copies_by_layer[i])
            for j, node in enumerate(layer):
                node.column = j

        return layers
