from collections.abc import Callable
from dataclasses import dataclass

import torch as t

from circuits.compile import LayeredGraph
from circuits.format import Bits


@dataclass(frozen=True, slots=True)
class SparseTensorGraph:
    weights: list[t.Tensor]
    biases: list[t.Tensor]
    zero_bias_default: bool = True
    biasless: bool = True

    @classmethod
    def from_graph(cls, graph: LayeredGraph) -> "SparseTensorGraph":
        weights, biases = cls._load_from_graph(graph)
        return cls(weights, biases)

    def __post_init__(self) -> None:
        # dtype = self.weights[0].dtype
        for i in range(len(self.biases)):
            
            if self.zero_bias_default:  # bias=0 is default not -1, it is sparser
                self.biases[i] += 1

            # if self.biasless:
            #     # folds the biases into the weights, assuming the first column is always 1
            #     print("b_len",self.biases[i].size(0)+1)
            #     bias_len = self.biases[i].size(0)

            #     # stack bias on top of weights:
            #     self.weights[i] = t.cat((self.biases[i], self.weights[i]), dim=0)

            #     col0 = t.zeros((1, bias_len+1), dtype=dtype)
            #     col0[0, 0] = 1  # propagates the hardcoded '1'
            #     self.weights[i] = t.cat((col0, self.weights[i]), dim=1)

            #     #  zero out biases, make it longer by 1
            #     self.biases[i] = t.zeros((bias_len+1), dtype=dtype)

    @staticmethod
    def _load_from_graph(
        graph: LayeredGraph, dtype: t.dtype = t.bfloat16
    ) -> tuple[list[t.Tensor], list[t.Tensor]]:
        weights: list[t.Tensor] = []
        biases: list[t.Tensor] = []

        for i, curr_nodes in enumerate(graph.layers[1:]):
            row_idx: list[int] = []
            col_idx: list[int] = []
            val_lst: list[int | float] = []
            for j, node in enumerate(curr_nodes):
                for synapse in node.synapses:
                    row_idx.append(j)
                    col_idx.append(synapse.column)
                    val_lst.append(synapse.weight)
            indices = t.tensor([row_idx, col_idx], dtype=t.long)
            values = t.tensor(val_lst, dtype=dtype)
            size = (len(curr_nodes), len(graph.layers[i]))
            weights.append(t.sparse_coo_tensor(indices, values, size, dtype=dtype))  # type: ignore
            bias = t.tensor([node.bias for node in curr_nodes], dtype=dtype)
            biases.append(bias)

        return weights, biases


class InitlessLinear(t.nn.Linear):
    """Skip init since all parameters will be specified"""

    def reset_parameters(self):
        pass


class StepMLP(t.nn.Module):
    """PyTorch MLP implementation with a step activation function"""

    def __init__(self, sizes: list[int], dtype: t.dtype = t.bfloat16):
        super().__init__()  # type: ignore
        self.dtype = dtype
        self.sizes = sizes
        mlp_layers = [
            InitlessLinear(in_s, out_s) for in_s, out_s in zip(sizes[:-1], sizes[1:])
        ]
        self.net = t.nn.Sequential(*mlp_layers).to(dtype)
        step_fn: Callable[[t.Tensor], t.Tensor] = lambda x: (x > 0).type(dtype)
        self.activation = step_fn
        self.n_sparse_params: int = sum(sizes[1:])  # add dense biases

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.type(self.dtype)
        for layer in self.net:
            x = self.activation(layer(x))
        return x

    def infer_bits(self, x: Bits) -> Bits:
        x_tensor = t.tensor(x.ints, dtype=self.dtype)
        with t.inference_mode():
            result = self.forward(x_tensor)
        result_ints = [int(el.item()) for el in t.IntTensor(result.int())]
        return Bits(result_ints)

    @classmethod
    def from_graph(cls, graph: LayeredGraph) -> "StepMLP":
        layer_sizes = [len(layer) for layer in graph.layers]
        sparse_graph = SparseTensorGraph.from_graph(graph)
        del graph
        mlp = cls(layer_sizes)
        mlp._load_from_sparse_graph(sparse_graph)
        return mlp

    def _load_from_sparse_graph(self, graph: SparseTensorGraph) -> None:
        for i, layer in enumerate(self.net):
            if not isinstance(layer, InitlessLinear):
                raise TypeError(f"Expected InitlessLinear, got {type(layer)}")
            layer.weight.data.copy_(graph.weights[i].to_dense())
            layer.bias.data.copy_(graph.biases[i])

    @property
    def n_params(self) -> str:
        n_dense = sum(p.numel() for p in self.parameters()) / 10**9
        n_sparse = self.n_sparse_params / 10**9
        return f"dense {n_dense:.2f}B, sparse {n_sparse:.2f}B"

    @property
    def layer_stats(self) -> str:
        res = f'layers: {len(self.sizes)}, max width: {max(self.sizes)}, widths: {self.sizes}\n'
        layer_n_params = [self.sizes[i]*self.sizes[i+1] for i in range(len(self.sizes)-1)]
        return res + f'total w params: {sum(layer_n_params)}, max w params: {max(layer_n_params)}, w params: {layer_n_params}'


import torch as t
import torch.nn as nn
import torch.nn.functional as F
t.set_default_dtype(t.bfloat16)


class SwiGLU(nn.Module):
    """SwiGLU (Swish-Gated Linear Unit) activation as used in modern transformers."""
    def __init__(self, in_features: int, out_features: int, dtype: t.dtype = t.bfloat16):
        super().__init__()  # type: ignore
        hidden_features = int(out_features * 2)
        self.w_silu = nn.Linear(in_features, hidden_features, bias=False)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_out = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.to(dtype=self.w_out.weight.dtype)
        return self.w_out(F.silu(self.w_silu(x)) * self.w_gate(x))


def get_swiglu_weights(w: t.Tensor, b: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Prepares SwiGLU weights from linear threshold circuit weight and bias matrix.
    1) Folds bias into weights by having feature at index 0 always be 1.
    2) Simulates a step fn with two offset ReLUs
    3) Simulates ReLU with SiLU by scaling up and down
    # Making two ReLUs a, b such that a-b is this fn:
    # y=0 until x=0.5-1/4c, then slope up until x=0.5+1/4c and y=1. Then y=1.
    # Demo: https://www.desmos.com/calculator/sk42yz8ami
    """
    c = 16  # making ReLU-simulated step fn steeper
    q = 16  # scaling before and after SiLU to avoid non-ReLU-like dip

    out_features = w.size(0) + 1
    # in_features = w.size(1) + 1

    # constructing for w_silu
    b1 = b - 0.5 - 1/(2*c)
    b2 = b - 0.5 + 1/(2*c)
    b1 = t.unsqueeze(b1.T, dim=1)
    b2 = t.unsqueeze(b2.T, dim=1)
    one = t.ones(1, 1)
    zeros = t.zeros(1, w.size(1))
    w1 = t.cat([
        t.cat([one, zeros], dim=1),
        t.cat([b1, w], dim=1),
        t.cat([one, zeros], dim=1),
        t.cat([b2, w], dim=1)
    ], dim=0)
    w1 *= c * q  # scale up
    w1[0,0] -= q  # to ensure that out vector begins with 1 

    # constructing for w_gate
    w2 = t.zeros_like(w1)
    w2[:,0] += 1  # gate = 1

    # constructing for w_out
    eye = t.eye(out_features)
    w3 = t.cat((-eye, eye), dim=1)
    w3 /= q  # scale down
    return w1, w2, w3


def swiglu_from_ltc(w: t.Tensor, b: t.Tensor) -> SwiGLU:
    swiglu = SwiGLU(w.size(1)+1, w.size(0)+1)
    for p, w in zip([swiglu.w_silu, swiglu.w_gate, swiglu.w_out],
                    get_swiglu_weights(w, b)):
        p.weight.data.zero_()
        p.weight.data[:w.size(0), :w.size(1)] = w
    return swiglu


def prep_input(x: list[int] | list[float] | list[bool], in_features: int):
    BOS = [1]
    padding = [0] * (in_features-len(x)-1)
    inp = BOS + x + padding
    return t.tensor(inp).unsqueeze(0)


def test_if_equal(a: t.Tensor, b: t.Tensor):
    if len(a) != len(b):
        print("len mismatch", len(a), len(b))
        return False
    for l, s in zip(a, b):
        if l != s:
            print("val mismatch", l, s)
            return False
    print("match!")


def get_ltc_vals(graph: LayeredGraph, i: int) -> list[float]:
    ltc_vals = [float(n.metadata['val']) for n in graph.layers[i]]
    return ltc_vals
