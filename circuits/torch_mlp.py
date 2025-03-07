from collections.abc import Callable

import torch as t

from circuits.graph import Graph
from circuits.format import Bits


class NoInitLinear(t.nn.Linear):
    """Skip init since all parameters will be specified"""

    def reset_parameters(self):
        pass


class StepMLP(t.nn.Module):
    """PyTorch MLP implementation with a step activation function"""

    def __init__(self, layer_sizes: list[int], dtype: t.dtype = t.bfloat16):
        super().__init__()  # type: ignore
        self.dtype = dtype
        self.sizes = layer_sizes
        mlp_layers = [
            NoInitLinear(in_s, out_s)
            for in_s, out_s in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.net = t.nn.Sequential(*mlp_layers).to(dtype)
        step_fn: Callable[[t.Tensor], t.Tensor] = lambda x: (x >= 0).type(dtype)
        self.activation = step_fn
        self.n_sparse_params: int = sum(layer_sizes[1:])  # add dense biases

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
    def from_graph(cls, graph: Graph) -> "StepMLP":
        layer_sizes = [len(layer) for layer in graph.layers]
        mlp = cls(layer_sizes)
        mlp._load_weights_from_graph(graph)
        return mlp

    def _load_weights_from_graph(self, graph: "Graph") -> None:
        for i, layer in enumerate(self.net):
            if not isinstance(layer, NoInitLinear):
                raise TypeError(f"Expected NoInitLinear, got {type(layer)}")
            curr_nodes = graph.layers[i + 1]
            row_idx: list[int] = []
            col_idx: list[int] = []
            val_list: list[int | float] = []
            for j, node in enumerate(curr_nodes):
                for parent in node.parents:
                    if parent.column is None:
                        raise ValueError("Parent node must have a column index")
                    row_idx.append(j)
                    col_idx.append(parent.column)
                    val_list.append(node.weights[parent])

            # Build a sparse tensor for this layer
            indices = t.tensor([row_idx, col_idx], dtype=t.long)
            values = t.tensor(val_list, dtype=self.dtype)
            size = (len(curr_nodes), layer.in_features)
            sparse_layer = t.sparse_coo_tensor(indices, values, size, dtype=self.dtype)  # type: ignore

            self.n_sparse_params += sparse_layer._nnz()
            weights = sparse_layer.to_dense()
            biases = t.tensor([node.bias for node in curr_nodes], dtype=self.dtype)

            # Copy the computed weights and biases into the layer
            layer.weight.data.copy_(weights)
            layer.bias.data.copy_(biases)

    @property
    def n_params(self) -> str:
        n_dense = sum(p.numel() for p in self.parameters()) / 10**9
        n_sparse = self.n_sparse_params / 10**9
        return f"dense {n_dense:.2f}B, sparse {n_sparse:.2f}B"
