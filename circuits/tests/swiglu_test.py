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
    # print("msg:", message.bitstr)

    graph = BlockGraph.compile(k.digest, len(message))
    # graph.print_activations()
    matrices = Matrices.from_blocks(graph)
    # print("matrices.mlist[0]", matrices.mlist[0], matrices.mlist[0].shape)
    mlp = swiglu_mlp_from_matrices(matrices)
    # print("mlp.layers[0].w_silu", mlp.layers[0].w_silu.weight.data, mlp.layers[0].w_silu.weight.data.shape)

    out = mlp.infer_bits(message)
    assert hashed.bitstr == out.bitstr, f"{hashed.bitstr} =/= {out.bitstr}"
    # expected = "0111111010"  # regression test
    # assert out.bitstr == expected

if __name__ == "__main__":

    # from circuits.utils.format import Bits
    # from circuits.utils.ftrace import find_instances
    # from circuits.neurons.core import Bit, Signal
    # b = Bits(10001)
    # found = find_instances(b, Bit)
    # print("found", found)
    # assert False

    test_mlp_swiglu_from_blocks()
