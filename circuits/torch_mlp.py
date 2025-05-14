from collections.abc import Callable
from dataclasses import dataclass

import torch as t

from circuits.compile import LayeredGraph
from circuits.format import Bits


@dataclass(frozen=True, slots=True)
class SparseTensorGraph:
    weights: list[t.Tensor]
    biases: list[t.Tensor]

    @classmethod
    def from_graph(cls, graph: LayeredGraph) -> "SparseTensorGraph":
        weights, biases = cls._load_from_graph(graph)
        return cls(weights, biases)

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
            biases.append(bias + 1)  # +1 so that sparser bias=0 is default, not -1
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
