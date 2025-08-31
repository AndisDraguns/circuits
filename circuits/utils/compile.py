from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from circuits.utils.graph import Graph, Level, Origin, Parent
from circuits.neurons.core import Bit
from circuits.utils.blocks import Block, BlockTracer, traverse
from circuits.utils.format import Bits
from circuits.utils.ftraceviz import visualize


@dataclass(frozen=True)
class BlockGraph(Graph):
    # TODO: reform as not a graph subclass but w .graph property?
    root: Block
    origin_blocks: list[list[Block]]

    @classmethod
    def compile(cls,
            function: Callable[..., list[Bit] | Bits],
            input_len: int,
            collapse: set[str] = set(),
            **kwargs: Any
            ) -> 'BlockGraph':
        """Compiles a function into a graph."""
        # {'__init__', 'outgoing', 'step'}
        tracer = BlockTracer(collapse)
        dummy_inp = Bits('0' * input_len)
        # dummy_inp = Bits('11001')
        # TODO: make it Bits/list[Bit]-agnostic
        # from circuits.utils.format import bitfun
        # function = bitfun(function)
        root = tracer.run(function, dummy_inp, **kwargs)
        visualize(root)
        origin_blocks = cls.set_origins(root)
        cls.set_narrow_origins(origin_blocks)
        levels = [Level(tuple([b.origin for b in level])) for level in origin_blocks]
        return cls(root = root, origin_blocks = origin_blocks, levels = tuple(levels))

    @staticmethod
    def set_origins(root: Block) -> list[list[Block]]:
        depth = root.top
        levels: list[list[Block]] = [[] for _ in range(depth)]

        # Set connections and add to levels
        for b in traverse(root):
            if b.name == 'gate':
                out = b.creation.data
                weights_in = out.source.weights
                bias = out.source.bias
                if 'untraced' in b.tags:
                    bias += b.origin.bias
            elif b.name == 'copy':
                weights_in = [1]
                bias = -1
            else:
                continue
            indices_in = [inp.creator.abs_x for inp in b.inputs if inp.creator is not None]
            incoming = [Parent(idx, int(w)) for idx, w in zip(indices_in, weights_in)]
            b.origin = Origin(b.abs_x, tuple(incoming), int(bias))
            levels[b.abs_y].append(b)

        # set correct w for connections to inputs
        for j, b in enumerate(levels[1]):
            # assumes that all levels[0] inputs are root inputs, and that other levels have no such inputs
            # without this, gates on this level have [], [] for indices and w
            origin = b.origin
            incoming = [Parent(inp.creator.abs_x, 1) for inp in b.inputs if inp.creator is not None]
            b.origin = Origin(origin.index, tuple(incoming), origin.bias)
        
        # set origins for inputs
        input_blocks: list[Block] = []
        assert len(levels[0]) == 0
        for b in traverse(root):
            if b.name == 'input':
                input_blocks.append(b)
        levels[0] = input_blocks
        for j, b in enumerate(levels[0]):
            b.origin = Origin(j, (), -1)

        # set origins for outputs
        for j, out in enumerate(root.outputs):
            b = out.creator
            assert b is not None
            incoming = [Parent(inp.creator.abs_x, 1) for inp in b.inputs if inp.creator is not None]
            b.origin = Origin(j, tuple(incoming), -1)
            levels[-1].append(b)

        if len(root.outputs) == 0:
            print("Warning, no outputs detected in the compiled function")
        if len(input_blocks) == 0:
            print("Warning, no inputs detected in the compiled function")

        return levels

    @staticmethod
    def set_narrow_origins(origin_blocks: list[list[Block]]) -> None:

        # record narrow indices
        to_narrow_index: dict[tuple[int, int], int] = dict()
        for i, level in enumerate(origin_blocks):
            for j, b in enumerate(level):
                to_narrow_index[(i, b.origin.index)] = j

        # set narrow indices
        for i, level in enumerate(origin_blocks[1:], start=1):  # skip input level
            for j, b in enumerate(level):
                origin = b.origin
                try:
                    index = to_narrow_index[(i, b.origin.index)]
                    incoming = [Parent(to_narrow_index[(i-1, p.index)], p.weight) for p in origin.incoming]
                except:
                    print(b.path)
                    assert False
                b.origin = Origin(index, tuple(incoming), origin.bias)

    def print_activations(self) -> None:
        for i, level in enumerate(self.origin_blocks):
            level_activations = [b.creation.data.activation for b in level]
            print(i, ''.join(str(int(a)) for a in level_activations))


    # def run(inputs: list[Bit]) -> list[Bit]:
    #     # save to block visualization
    #     pass
