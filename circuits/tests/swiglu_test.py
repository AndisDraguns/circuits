from circuits.utils.compile import BlockGraph
from circuits.examples.keccak import Keccak
from circuits.dense.swiglu import swiglu_mlp_from_matrices
from circuits.dense.matrices import Matrices

def test_mlp_swiglu_from_blocks():
    """Test SwigLU MLP obtained from blocks"""
    k = Keccak(c=10, l=0, n=3, pad_char="_")   # reduced number of rounds for testing
    phrase = "Rachmaninoff"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)

    graph = BlockGraph.compile(k.digest, len(message))
    matrices = Matrices.from_blocks(graph)
    mlp = swiglu_mlp_from_matrices(matrices)

    out = mlp.infer_bits(message)
    assert hashed.bitstr == out.bitstr, f"{hashed.bitstr} =/= {out.bitstr}"
    expected = "10001"  # regression test
    assert out.bitstr == expected

if __name__ == "__main__":
    test_mlp_swiglu_from_blocks()
