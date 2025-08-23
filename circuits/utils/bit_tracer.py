from dataclasses import dataclass

from circuits.neurons.core import Bit
from circuits.utils.ftraceviz import Tracer, visualize
from circuits.utils.blocks import Block, traverse


@dataclass
class BitTracer(Tracer[Bit]):
    use_defaults: bool = False
    def __post_init__(self):
        self.tracked_type = Bit
        self.formatter = lambda x: str(int(x.activation))
        if self.use_defaults:
            c = {'__init__', '__post_init__', '<lambda>'}
            c |= {'outgoing', 'const', 'xor', 'inhib', 'step'}
            c |= {'format', 'bitlist', '_bitlist_from_value', '_is_bit_list', 'from_str'}
            c |= {'_bitlist_to_msg', 'msg_to_state', 'get_round_constants', 'get_functions'}
            c |= {'lanes_to_state', 'state_to_lanes', 'get_empty_lanes', 'copy_lanes'}
            c |= {'rho_pi', 'rot', 'reverse_bytes'}
            self.collapse |= c



if __name__ == '__main__':
    from circuits.utils.format import Bits
    from circuits.examples.keccak import Keccak
    def f(m: Bits, k: Keccak) -> list[Bit]:
        return k.digest(m).bitlist
    k = Keccak(c=10, l=0, n=2, pad_char='_')
    tracer = BitTracer(collapse = {'__init__', 'outgoing', 'step'})
    msg1 = k.format("Reify semantics as referentless embeddings", clip=True)
    b1 = tracer.run(f, m=msg1, k=k)
    # msg2 = k.format("Test", clip=True)
    # b2 = tracer.run(f, m=msg2, k=k)
    # tracer.mark_differences(b1, b2)
    visualize(b1)


# def compiled_from_io(inputs: list[Signal], outputs: list[Signal]) -> Graph:
#     """Compiles a graph for function f using dummy input and output=f(input)."""
#     return Graph(inputs, outputs)

from typing import Any
from collections.abc import Callable
from circuits.neurons.core import Signal, const

from circuits.utils.format import Bits 
def compile(function: Callable[..., list[Signal]], input_len: int, **kwargs: Any) -> Block[Bit]:
    """Compiles a function into a graph."""
    tracer = BitTracer(collapse = {'__init__', 'outgoing', 'step'})
    inp = Bits(const("0" * input_len))
    # inp = Bits('0101001')
    # out = function(inp, **kwargs)
    root = tracer.run(function, inp, **kwargs)
    return root

    # return compiled_from_io(inp, out)


from circuits.utils.blocks import Flow
from circuits.utils.misc import OrderedSet
def create_output_blocks(root: Block[Bit]) -> list[Block[Bit]]:
    res = []
    for j, out in enumerate(root.outputs):
        b = Block[Bit]('output', 'output-{j}', flavour='output', abs_x=j, abs_y=root.abs_y+1)
        b.inputs = OrderedSet([Flow[Bit](out.data, b, creator=out.creator, prev=out)])
        b.outputs = OrderedSet([Flow[Bit](out.data, b, creator=b)])
        res.append(b)
    return res


import torch as t
import torch.nn as nn
import torch.nn.functional as F


def b_info_layer_to_params(layer: list[dict[str, Any]], layer_shape: tuple[int, int], dtype: t.dtype, debias: bool = True
                    ) -> tuple[t.Tensor, t.Tensor]:
    """Convert layer to a sparse weight matrix and dense bias matrix"""
    row_idx: list[int] = []
    col_idx: list[int] = []
    val_lst: list[int | float] = []
    for b_info in layer:
        for p_idx, p_w in zip(b_info['parent_indices'], b_info['parent_weights']):
            row_idx.append(b_info['abs_x'])
            col_idx.append(p_idx)
            val_lst.append(p_w)
    indices = t.tensor([row_idx, col_idx], dtype=t.long)
    values = t.tensor(val_lst, dtype=dtype)
    # size = (len(layer), size_in)
    size = layer_shape
    w_sparse = t.sparse_coo_tensor(indices, values, size, dtype=dtype)  # type: ignore
    b_placeholder = [-1] * layer_shape[0]
    # print("layer_shape =", layer_shape, "abs_xs =",[b_info['abs_x'] for b_info in layer])
    for b_info in layer:
        b_placeholder[b_info['abs_x']] = b_info['bias']
    b = t.tensor(b_placeholder, dtype=dtype)
    if debias:
        b += 1
    # TODO: sparse biases

    return w_sparse, b


def get_block_info_for_mlp(root: Block[Bit]) -> tuple[list[list[dict[str, Any]]], list[tuple[int, int]]]:
    depth = root.top+1  # +1 for including inputs and outputs
    levels = [[] for _ in range(depth)]
    for b in traverse(root):
        if b.name not in {'gate', 'copy'}:
            continue
        # if b.name == 'gate' and 'constant' in b.tags:
            # b_info = {'b': b, 'abs_x': b.abs_x, 'parent_indices': [0]}
        b_info = {'b': b, 'abs_x': b.abs_x, 'parent_indices': [inp.creator.abs_x for inp in b.inputs]}
        if b.name == 'gate':
            out = b.children[0].creation.data
            b_info['parent_weights'] = out.source.weights
            b_info['bias'] = out.source.bias
        if b.name == 'copy':
            b_info['parent_weights'] = [1]
            b_info['bias'] = -1
        levels[b.abs_y] += [b_info]
    
    # set correct w for connections to inputs
    for j, b_info in enumerate(levels[0]):
        if len(b_info['b'].inputs)>0:
            b_info['parent_weights'] = [1]
            b_info['parent_indices'] = [list(b_info['b'].inputs)[0].creator.abs_x]

    # set correct w for connections from outputs
    output_blocks = create_output_blocks(root)
    # print("len(levels[-1])", len(levels[-1]))
    for b in output_blocks:
        parent_indices = [inp.creator.abs_x for inp in b.inputs]
        b_info = {'b': b, 'abs_x': b.abs_x, 'parent_indices': parent_indices, 'parent_weights': [1], 'bias': -1}
        # print(parent_indices)
        # print([f"{inp.creator.abs_x}, {inp.creator.path}" for inp in b.inputs])
        levels[-1].append(b_info)
    # assert False


    # print("levels[-2]:")
    # for j, b_info in enumerate(levels[-2]):
    #     print(j, b_info['parent_indices'], b_info['parent_indices'])
    #     # print([f"{inp.creator.abs_x}, {inp.creator.path}" for inp in b.inputs])
    # assert False


    input_w = len(root.inputs)
    middle_w = root.right
    output_w = len(root.outputs)
    # sizes_in = [input_w] + [middle_w]*(depth-1)
    layer_shapes = [(middle_w, middle_w) for _ in range(depth)]
    layer_shapes[0] = (middle_w, input_w)
    # layer_shapes = [(x,y) for x, y in zip(sizes_in[:-1], sizes_in[1:])]
    layer_shapes[-1] = (output_w, middle_w)
    assert len(levels) == len(layer_shapes), f"{len(levels)} {len(layer_shapes)}"

    # for i, level in enumerate(levels):
    #     activations = [0] * layer_shapes[i][0]
    #     for el in level:
    #         b = el['b']
    #         assert len(activations) > b.abs_x, f"{len(activations)} {b.abs_x} {b.path}"
    #         activations[b.abs_x] = int(list(b.outputs)[0].data.activation)
    #     print(f"level {i} gate outputs:\t", Bits(activations[:]))
        # print(f"level {i} gate outputs:\t", Bits([int(list(el['b'].outputs)[0].data.activation) for el in level][:60]))
        # for b_info in level:

    return levels, layer_shapes
