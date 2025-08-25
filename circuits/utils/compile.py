from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from circuits.utils.graph import Graph, Level, Origin, Parent
from circuits.neurons.core import Bit
from circuits.utils.blocks import Block, traverse
from circuits.utils.bit_tracer import BitTracer
from circuits.utils.format import Bits
# from circuits.utils.format import bitfun


@dataclass(frozen=True)
class BlockGraph(Graph):
    # TODO: reform as not a graph subclass but w .graph property
    root: Block
    origin_blocks: list[list[Block]]

    @classmethod
    def compile(cls, function: Callable[..., list[Bit]], input_len: int, **kwargs: Any) -> Graph:
        """Compiles a function into a graph."""
        tracer = BitTracer(collapse = {'__init__', 'outgoing', 'step'})
        dummy_inp = Bits('0' * input_len)
        # bitfun(function)
        root = tracer.run(function, dummy_inp, **kwargs)
        origin_blocks = cls.set_origins(root)
        cls.set_narrow_origins(origin_blocks)
        levels = [Level([b.info['origin'] for b in level]) for i, level in enumerate(origin_blocks)]
        return cls(root = root, origin_blocks = origin_blocks, levels = levels)

    @staticmethod
    def set_origins(root: Block) -> list[list[Block]]:
        depth = root.top + 2  # +2 for including inputs and outputs
        levels = [[] for _ in range(depth)]

        # Set connections and add to levels
        for b in traverse(root):
            if b.name == 'gate':
                out = b.children[0].creation.data
                weights_in = out.source.weights
                bias = out.source.bias
            elif b.name == 'copy':
                weights_in = [1]
                bias = -1
            else:
                continue
            indices_in = [inp.creator.abs_x for inp in b.inputs]
            incoming = [Parent(idx, w) for idx, w in zip(indices_in, weights_in)]
            b.info['origin'] = Origin(b.abs_x, incoming, bias)
            levels[b.abs_y+1].append(b)

        # set correct w for connections to inputs
        for j, b in enumerate(levels[1]):
            # assumes that all levels[0] inputs are root inputs, and that other levels have no such inputs
            # without this, gates on this level have [], [] for indices and w 
            origin = b.info['origin']
            # origin.parent_indices = [inp.creator.abs_x for inp in b.inputs]
            # origin.parent_weights = [1]*len(b.inputs)
            incoming = [Parent(inp.creator.abs_x, 1) for inp in b.inputs]
            b.info['origin'] = Origin(origin.index, incoming, origin.bias)
        
        # set origins for inputs
        levels[0] = [inp.creator for inp in root.inputs]
        for j, b in enumerate(levels[0]):
            b.info['origin'] = Origin(j, [], -1)


        # create output blocks
        from circuits.utils.blocks import Flow
        from circuits.utils.misc import OrderedSet
        for j, out in enumerate(root.outputs):
            b = Block[Bit]('output', 'output-{j}', flavour='output', abs_x=j, abs_y=root.abs_y+1)
            b.inputs = OrderedSet([Flow[Bit](out.data, b, creator=out.creator, prev=out)])
            b.outputs = OrderedSet([Flow[Bit](out.data, b, creator=b)])
            incoming = [Parent(inp.creator.abs_x, 1) for inp in b.inputs]
            b.info['origin'] = Origin(j, incoming, -1)
            levels[-1].append(b)

        return levels

    @staticmethod
    def set_narrow_origins(origin_blocks: list[list[Block]]) -> None:
        to_narrow_index: dict[tuple[int, int], int] = dict()
        for i, level in enumerate(origin_blocks):
            for j, b in enumerate(level):
                # try:
                to_narrow_index[(i, b.info['origin'].index)] = j
                #     if i == 0:
                #         print(i, j, b.info['origin'].index)
                # except:
                #     print(b.path)
                #     assert False
        for i, level in enumerate(origin_blocks[1:], start=1):  # skip input level
            for j, b in enumerate(level):
                origin = b.info['origin']
                # try:
                index = to_narrow_index[(i, b.info['origin'].index)]
                incoming = [Parent(to_narrow_index[(i-1, p.index)], p.weight) for p in origin.incoming]
                # except:
                #     print(b.path)
                #     assert False
                b.info['origin'] = Origin(index, incoming, origin.bias)


    # def get_levels(origin_blocks: list[list[Block[Bit]]]) -> list[Level]:
    #     # widths = [len(level) for level in origin_blocks]
    #     # shapes = [(out_w, inp_w) for out_w, inp_w in zip(widths[1:], widths[:-1])]
    #     levels = [Level([b.info['origin'] for b in level]) for i, level in enumerate(origin_blocks)]


    # def run(inputs: list[Bit]) -> list[Bit]:
    #     # save to block visualization
    #     pass

    # middle_w = self.root.right
    # output_w = len(self.root.outputs)
    # layer_shapes = [(middle_w, middle_w) for _ in range(depth)]
    # layer_shapes[0] = (middle_w, (self.root.inputs))
    # layer_shapes[-1] = (output_w, middle_w)
    # assert len(self.levels) == len(layer_shapes)
    # return layer_shapes

    # def origin_blocks_to_layers_wide(width: int, origin_blocks: list[list[Block[Bit]]]) -> list[Layer]:
    #     layers = []
    #     for level in origin_blocks:
    #         origins = [Origin(j, [], [], -1) for j in range(width)]
    #         for b in level:
    #             origins[b.info['origin'].index] = b.info['origin']
    #         layers.append(Layer(origins))
    #     # remove unneeded width:
    #     layers[0] = Layer([b.info['origin'] for b in origin_blocks[0]])
    #     layers[-1] = Layer([b.info['origin'] for b in origin_blocks[-1]])
    #     return layers