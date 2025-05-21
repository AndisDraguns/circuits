from collections.abc import Callable
from dataclasses import dataclass

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from circuits.compile import Graph, Node
from circuits.format import Bits



@dataclass(frozen=True, slots=True)
class Parameters:
    weights: list[t.Tensor]  # sparse (h, w)
    biases: list[t.Tensor]  # (1, h)

    @classmethod
    def from_graph(cls, graph: Graph, dtype: t.dtype) -> "Parameters":
        """
        Set parameters of the model from weights and biases.
        Debias adds 1 to biases, shifting the default bias from -1 to sparser 0.
        LTC default bias is -1.
        """
        layers = graph.layers[1:]  # skip input layer as it has no incoming weights
        sizes_in = [len(l) for l in graph.layers]  # incoming weight sizes
        params = [cls.layer_to_params(l, s, dtype) for l, s in zip(layers, sizes_in)]
        weights, biases = zip(*params)
        return cls(list(weights), list(biases))

    @staticmethod
    def layer_to_params(layer: list[Node], size_in: int, dtype: t.dtype, debias: bool = True
                        ) -> tuple[t.Tensor, t.Tensor]:
        row_idx: list[int] = []
        col_idx: list[int] = []
        val_lst: list[int | float] = []
        for j, node in enumerate(layer):
            for p in node.parents:
                row_idx.append(j)
                col_idx.append(p.column)
                val_lst.append(node.weights[p])
        indices = t.tensor([row_idx, col_idx], dtype=t.long)
        values = t.tensor(val_lst, dtype=dtype)
        size = (len(layer), size_in)
        w_sparse = t.sparse_coo_tensor(indices, values, size, dtype=dtype)  # type: ignore
        b = t.tensor([node.bias for node in layer], dtype=dtype)
        if debias:
            b += 1
        # TODO: sparse biases
        return w_sparse, b

    @staticmethod
    def fold_bias(w: t.Tensor, b: t.Tensor) -> t.Tensor:
        """Folds bias into weights, assuming input feature 0 is always 1."""
        one = t.ones(1, 1)
        zeros = t.zeros(1, w.size(1))
        # assumes row vector bias that is transposed during forward pass
        bT = t.unsqueeze(b, dim=-1)
        wb = t.cat([
            t.cat([one, zeros], dim=1),
            t.cat([bT, w], dim=1),
        ], dim=0)
        return wb

    @property
    def matrices(self) -> list[t.Tensor]:
        """Creates dense weight matrices with biases folded in."""
        dense_w = [w.to_dense() for w in self.weights]
        # print("dense_w", dense_w[0].shape) # type: ignore
        wb = [self.fold_bias(w, b) for w, b in zip(dense_w, self.biases)]
        # print("wb", wb[0].shape) # type: ignore
        return wb




@dataclass(frozen=True, slots=True)
class Instruction:
    """Instruction for a single matrix multiplication. Values are in binary.
    flat = as a 1D vector.
    struct = as a 3D structure."""
    flat: t.Tensor
    struct: t.Tensor
    out_features: int
    in_features: int

    @classmethod
    def from_struct(cls, struct: t.Tensor) -> "Instruction":
        h, w, d = struct.shape
        assert d == 2
        flat = struct.view(h*w*2)
        return cls(flat, struct, h, w)
    
    @classmethod
    def from_flat(cls, flat: t.Tensor, out_features: int, in_features: int) -> "Instruction":
        """Convert flat list to matrix"""
        struct = flat.view(out_features, in_features, 2)
        return cls(flat, struct, out_features, in_features)

    @property
    def flatstr(self) -> str:
        ints = [int(el.item()) for el in self.flat]
        return ''.join([str(el) for el in ints])

    def __repr__(self) -> str:
        return f"Instruction({self.flatstr})"



@dataclass(frozen=True, slots=True)
class Matrices:
    mlist: list[t.Tensor]
    dtype: t.dtype = t.int

    @classmethod
    def from_graph(cls, graph: Graph, dtype: t.dtype=t.int) -> "Matrices":
        params = Parameters.from_graph(graph, dtype=dtype) 
        return cls(params.matrices, dtype=dtype)

    @staticmethod
    def ternarize_matrix(m: t.Tensor, fwidths: list[int], next_fwidths: list[int]) -> t.Tensor:
        """
        Ternarize int matrix with max abs value per column
        m: (h, w)
        fwidths: (w,)
        next_fwidths: (w-1,)
        """
        m_wide: list[t.Tensor] = []
        for j in range(m.size(1)):
            fw = fwidths[j]
            col = m[:, j]
            indices = t.arange(fw).expand(col.size(0), fw)
            abs_val = t.abs(col).unsqueeze(1)
            signs = t.sign(col).unsqueeze(1)
            col_wide = t.where(indices < abs_val, signs, t.zeros_like(indices))
            m_wide.append(col_wide)
        m_ternary = t.repeat_interleave(t.cat(m_wide, dim=1), t.tensor(next_fwidths), dim=0)
        return m_ternary

    @staticmethod
    def encode_binary_matrix(m: t.Tensor) -> t.Tensor:
        """Encode a ternary matrix as a structure with binary values"""
        h, w = m.shape
        result = t.zeros((h, w, 2), dtype=t.int)  # For 0, set both positions to 0 (default)
        # result[m==-1, 1] = 1  # For -1, set second position to 1
        # result[m==1, 0] = 1   # For 1, set first position to 1
        neg1_indices = (m == -1).nonzero(as_tuple=True)
        pos1_indices = (m == 1).nonzero(as_tuple=True)
        result[neg1_indices[0], neg1_indices[1], 1] = 1
        result[pos1_indices[0], pos1_indices[1], 0] = 1
        return result


    @property
    def ternarized(self) -> list[t.Tensor]:
        """Convert matrix elements from int to [-1, 0, 1] while maintaining the functionality.
        
        1) First we expand each column by repeating the sign up to the max abs value in that column.
        Assuming that the input features are also repeated accordingly, the result is the same. 
        
        2) Then we repeat the rows according to the next matrix's max abs col values.
        This is done to ensure that the output features are repeated correctly for the next matrix.

        Here's an example:
        -2 1  0      -1 -1 0  1   0  0      -1 -1 0  1   0  0
        3  1 -2  ->   1  1 1  1  -1 -1  ->   1  1 1  1  -1 -1
                                             1  1 1  1  -1 -1
        Here for the second step we assumed that the next matrix has these max abs col values [1, 2].
        """
        ms = [m.int() for m in self.mlist]

        def max_abs_col(m: t.Tensor) -> list[int]:
            """Calculate max abs value per column
            # [0] to get values from (values, indices) tuple"""
            return m.abs().max(dim=0)[0].int().tolist()  # type: ignore

        # calculate feature widths for each col in each matrix:
        fwidths = [max_abs_col(m) for m in ms]
        out_size = self.mlist[-1].size(0)
        fwidths += [[1] for _ in range(out_size)]  # last next_fwidths is 1s, i.e. unchanged

        # ternarize each matrix
        args = zip(ms, fwidths, fwidths[1:])
        m_ternary = [self.ternarize_matrix(m, fw1, fw2) for m, fw1, fw2 in args]

        # adaptor matrix expands the input vector so that it can be used with matrices_ternary
        eye = t.eye(self.mlist[0].size(1), dtype=m_ternary[0].dtype)
        first_fwidths = t.tensor(fwidths[0])
        adaptor = t.repeat_interleave(eye, first_fwidths, dim=0)

        # adds the adaptor matrix to the beginning of the sequence for automatic conversion
        matrix_sequence = [adaptor] + m_ternary
        return matrix_sequence


    @property
    def encoded(self) -> list[t.Tensor]:
        """Encode matrices in binary"""
        return [self.encode_binary_matrix(m) for m in self.ternarized]


    @property
    def instructions(self) -> list[Instruction]:
        """Convert matrices to instructions"""
        instructions = [Instruction.from_struct(s) for s in self.encoded]
        return instructions







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
            # print("In MLP:",Bits([int(el.item()) for el in x]))
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
        weights, biases = cls.params_from_graph(graph, dtype=mlp.dtype)
        mlp.load_params(weights, biases)
        print("layer_sizes:", layer_sizes)
        return mlp

    def load_params(self, weights: list[t.Tensor], biases: list[t.Tensor]) -> None:
        for i, layer in enumerate(self.net):
            if not isinstance(layer, InitlessLinear):
                raise TypeError(f"Expected InitlessLinear, got {type(layer)}")
            layer.weight.data.copy_(weights[i].to_dense())
            layer.bias.data.copy_(biases[i])

    @staticmethod
    def params_from_graph(graph: Graph, dtype: t.dtype = t.bfloat16
    ) -> tuple[list[t.Tensor], list[t.Tensor]]:
        params = Parameters.from_graph(graph, dtype=dtype)
        return params.weights, params.biases

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








# step = lambda x: t.threshold(x, 0, 1)
class MLP(nn.Module):
    """PyTorch MLP implementation"""

    def __init__(self, sizes: list[int], activation: nn.Module, dtype: t.dtype):
        super().__init__()  # type: ignore
        self.dtype = dtype
        layers: list[nn.Module] = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size, dtype=dtype))
            layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x.to(self.dtype))








# t.set_default_dtype(t.bfloat16)
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


def get_ltc_vals(graph: Graph, i: int) -> list[float]:
    ltc_vals = [float(n.metadata['val']) for n in graph.layers[i]]
    return ltc_vals
