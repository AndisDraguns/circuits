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
    # out = function(inp, **kwargs)
    root = tracer.run(function, inp, **kwargs)
    return root

    # return compiled_from_io(inp, out)



import torch as t
import torch.nn as nn
import torch.nn.functional as F


def b_info_layer_to_params(layer: list[dict[str, Any]], size_in: int, dtype: t.dtype, debias: bool = True
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
    size = (len(layer), size_in)
    w_sparse = t.sparse_coo_tensor(indices, values, size, dtype=dtype)  # type: ignore
    b_placeholder = [-1]*size_in
    for b_info in layer:
        b_placeholder[b_info['abs_x']] = b_info['bias']
    b = t.tensor(b_placeholder, dtype=dtype)
    if debias:
        b += 1
    # TODO: sparse biases
    return w_sparse, b


def get_block_info_for_mlp(root: Block[Bit]) -> list[list[dict[str, Any]]]:
    depth = root.top+1
    levels = [[] for _ in range(depth)]
    for b in traverse(root):
        if b.name not in {'gate', 'copy'}:
            continue
        b_info = {'b': b}
        b_info = {'abs_x': b.abs_x}
        b_info['parent_indices'] = [inp.creator.abs_x for inp in b.inputs]
        if b.name == 'gate':
            out = b.children[0].creation.data
            b_info['parent_weights'] = out.source.weights
            b_info['bias'] = out.source.bias
        if b.name == 'copy':
            b_info['parent_weights'] = [1]
            b_info['bias'] = -1
        levels[b.abs_y+1].append(b_info)  # +1 as block returns from top
    return levels