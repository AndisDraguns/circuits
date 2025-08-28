from collections.abc import Callable

import torch as t

from circuits.sparse.compile import Graph
from circuits.utils.format import Bits
from circuits.utils.compile import BlockGraph
from circuits.dense.matrices import Matrices



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
            InitlessLinear(in_s, out_s, bias=False) for in_s, out_s in zip(sizes[:-1], sizes[1:])
        ]
        self.net = t.nn.Sequential(*mlp_layers).to(dtype)
        step_fn: Callable[[t.Tensor], t.Tensor] = lambda x: (x > 0).type(dtype)
        self.activation_fn = step_fn

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.type(self.dtype)
        for layer in self.net:
            x = self.activation_fn(layer(x))
        return x
        # for i, layer in enumerate(self.net):
            # print(i, Bits(list(x.int().tolist())))  # type: ignore
            # x = self.activation(layer(x))
        # print(len(self.net), Bits(list(x.int().tolist())))  # type: ignore
        # return x

    def infer_bits(self, x: Bits, auto_constant: bool = True) -> Bits:
        if auto_constant:
            x = Bits('1') + x
        x_tensor = t.tensor(x.ints, dtype=self.dtype)
        with t.inference_mode():
            result = self.forward(x_tensor)
        result_ints = [int(el.item()) for el in t.IntTensor(result.int())]
        if auto_constant:
            result_ints = result_ints[1:]
        return Bits(result_ints)

    @classmethod
    def from_graph(cls, graph: Graph) -> "StepMLP":
        matrices = Matrices.from_graph(graph)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        return mlp

    @classmethod
    def from_blocks(cls, graph: BlockGraph) -> "StepMLP":
        matrices = Matrices.from_blocks(graph)
        mlp = cls(matrices.sizes)
        mlp.load_params(matrices.mlist)
        return mlp

    def load_params(self, weights: list[t.Tensor]) -> None:
        for i, layer in enumerate(self.net):
            if not isinstance(layer, InitlessLinear):
                raise TypeError(f"Expected InitlessLinear, got {type(layer)}")
            layer.weight.data.copy_(weights[i].to_dense())

    @property
    def n_params(self) -> str:
        n_dense = sum(p.numel() for p in self.parameters()) / 10**9
        return f"{n_dense:.2f}B"

    @property
    def layer_stats(self) -> str:
        res = f'layers: {len(self.sizes)}, max width: {max(self.sizes)}, widths: {self.sizes}\n'
        layer_n_params = [self.sizes[i]*self.sizes[i+1] for i in range(len(self.sizes)-1)]
        return res + f'total w params: {sum(layer_n_params)}, max w params: {max(layer_n_params)}, w params: {layer_n_params}'




# class MLP(nn.Module):
#     """Simple PyTorch MLP implementation"""

#     def __init__(self, sizes: list[int], activation: nn.Module, dtype: t.dtype):
#         super().__init__()  # type: ignore
#         self.dtype = dtype
#         layers: list[nn.Module] = []
#         for in_size, out_size in zip(sizes[:-1], sizes[1:]):
#             layers.append(nn.Linear(in_size, out_size, dtype=dtype))
#             layers.append(activation)
#         self.net = nn.Sequential(*layers)

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         return self.net(x.to(self.dtype))




# # t.set_default_dtype(t.bfloat16)
# class SwiGLU(nn.Module):
#     """SwiGLU (Swish-Gated Linear Unit) activation as used in modern transformers."""
#     def __init__(self, in_features: int, out_features: int, dtype: t.dtype = t.bfloat16):
#         super().__init__()  # type: ignore
#         self.dtype = dtype
#         self.in_features = in_features
#         self.out_features = out_features
#         hidden_features = int(out_features * 2)
#         self.w_silu = nn.Linear(in_features, hidden_features, bias=False)
#         self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
#         self.w_out = nn.Linear(hidden_features, out_features, bias=False)

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         x = x.to(dtype=self.w_out.weight.dtype)
#         return self.w_out(F.silu(self.w_silu(x)) * self.w_gate(x))



# def get_swiglu_weights(w: t.Tensor, b: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
#     """
#     Prepares SwiGLU weights from linear threshold circuit weight and bias matrix.
#     1) Folds bias into weights by having feature at index 0 always be 1.
#     2) Simulates a step fn with two offset ReLUs
#     3) Simulates ReLU with SiLU by scaling up and down
#     # Making two ReLUs a, b such that a-b is this fn:
#     # y=0 until x=0.5-1/4c, then slope up until x=0.5+1/4c and y=1. Then y=1.
#     # Demo: https://www.desmos.com/calculator/sk42yz8ami
#     """
#     c = 16  # making ReLU-simulated step fn steeper
#     q = 16  # scaling before and after SiLU to avoid non-ReLU-like dip

#     out_features = w.size(0) + 1

#     # constructing for w_silu
#     b1 = b - 0.5 - 1/(2*c)
#     b2 = b - 0.5 + 1/(2*c)
#     b1 = t.unsqueeze(b1.T, dim=1)
#     b2 = t.unsqueeze(b2.T, dim=1)
#     one = t.ones(1, 1)
#     zeros = t.zeros(1, w.size(1))
#     w1 = t.cat([
#         t.cat([one, zeros], dim=1),
#         t.cat([b1, w], dim=1),
#         t.cat([one, zeros], dim=1),
#         t.cat([b2, w], dim=1)
#     ], dim=0)
#     w1 *= c * q  # scale up
#     w1[0,0] -= q  # to ensure that out vector begins with 1 

#     # constructing for w_gate
#     w2 = t.zeros_like(w1)
#     w2[:,0] += 1  # gate = 1

#     # constructing for w_out
#     eye = t.eye(out_features)
#     w3 = t.cat((-eye, eye), dim=1)
#     w3 /= q  # scale down
#     return w1, w2, w3


# def swiglu_from_ltc(w: t.Tensor, b: t.Tensor) -> SwiGLU:
#     swiglu = SwiGLU(w.size(1)+1, w.size(0)+1)
#     for p, w in zip([swiglu.w_silu, swiglu.w_gate, swiglu.w_out],
#                     get_swiglu_weights(w, b)):
#         p.weight.data.zero_()
#         p.weight.data[:w.size(0), :w.size(1)] = w
#     return swiglu


# def prep_input(x: list[int] | list[float] | list[bool], in_features: int):
#     BOS = [1]
#     padding = [0] * (in_features-len(x)-1)
#     inp = BOS + x + padding
#     return t.tensor(inp).unsqueeze(0)


# def test_if_equal(a: t.Tensor, b: t.Tensor):
#     if len(a) != len(b):
#         print("len mismatch", len(a), len(b))
#         return False
#     for l, s in zip(a, b):
#         if l != s:
#             print("val mismatch", l, s)
#             return False
#     print("match!")


# def get_ltc_vals(graph: Graph, i: int) -> list[float]:
#     ltc_vals = [float(n.metadata['val']) for n in graph.layers[i]]
#     return ltc_vals



# class MLPSwiGLU(nn.Module):
#     """SwiGLU MLP for training on binary classification tasks."""
    
#     def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, 
#                  dtype: t.dtype = t.float32):
#         super().__init__()  # type: ignore
#         self.dtype = dtype
        
#         # Build layers
#         layers: list[SwiGLU | nn.Linear] = []
#         prev_size = input_size
        
#         # Hidden layers with SwiGLU activation
#         for hidden_size in hidden_sizes:
#             layers.append(SwiGLU(prev_size, hidden_size, dtype=dtype))
#             prev_size = hidden_size
#         layers.append(nn.Linear(prev_size, output_size, dtype=dtype))
#         # layers.append(SwiGLU(prev_size, output_size, dtype=dtype))
#         self.layers: nn.Sequential = nn.Sequential(*layers)
    
#     def forward(self, x: t.Tensor) -> t.Tensor:
#         return self.layers(x.to(self.dtype))

#     def predict(self, x: t.Tensor) -> t.Tensor:
#         """Binary prediction with sigmoid + threshold."""
#         logits = self.forward(x)
#         return (t.sigmoid(logits) > 0.5).float()