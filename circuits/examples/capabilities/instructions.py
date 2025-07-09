from dataclasses import dataclass
from circuits.dense.mlp import Matrices
import torch as t


@dataclass(frozen=True, slots=True)
class Instruction:
    """Instruction for a single matrix multiplication.
    Metagate takes an instruction as activations and multiplies it with an input vector.
    Values are in binary.
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


def get_instructions(matrices: Matrices) -> list[Instruction]:
    """Get instructions from matrices"""
    ternarized_matrices = ternarize_matrices(matrices)
    binary_structs = [matrix_to_struct(m) for m in ternarized_matrices]
    return [Instruction.from_struct(s) for s in binary_structs]


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


def ternarize_matrices(matrices: Matrices) -> list[t.Tensor]:
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
    mlist = matrices.mlist
    ms = [m.int() for m in mlist]

    def max_abs_col(m: t.Tensor) -> list[int]:
        """Calculate max abs value per column
        # [0] to get values from (values, indices) tuple"""
        return m.abs().max(dim=0)[0].int().tolist()  # type: ignore

    # calculate feature widths for each col in each matrix:
    fwidths = [max_abs_col(m) for m in ms]
    out_size = mlist[-1].size(0)
    fwidths += [[1] for _ in range(out_size)]  # last next_fwidths is 1s, i.e. unchanged

    # ternarize each matrix
    args = zip(ms, fwidths, fwidths[1:])
    m_ternary = [ternarize_matrix(m, fw1, fw2) for m, fw1, fw2 in args]

    # adaptor matrix expands the input vector so that it can be used with matrices_ternary
    eye = t.eye(mlist[0].size(1), dtype=m_ternary[0].dtype)
    first_fwidths = t.tensor(fwidths[0])
    adaptor = t.repeat_interleave(eye, first_fwidths, dim=0)

    # adds the adaptor matrix to the beginning of the sequence for automatic conversion
    matrix_sequence = [adaptor] + m_ternary
    return matrix_sequence


def matrix_to_struct(m: t.Tensor) -> t.Tensor:
    """Encode a ternary matrix as a 3D structure with binary values"""
    h, w = m.shape
    result = t.zeros((h, w, 2), dtype=t.int)  # For 0, set both positions to 0 (default)
    # result[m==-1, 1] = 1  # For -1, set second position to 1
    # result[m==1, 0] = 1   # For 1, set first position to 1
    neg1_indices = (m == -1).nonzero(as_tuple=True)
    pos1_indices = (m == 1).nonzero(as_tuple=True)
    result[neg1_indices[0], neg1_indices[1], 1] = 1
    result[pos1_indices[0], pos1_indices[1], 0] = 1
    return result  # (h, w, 2)
