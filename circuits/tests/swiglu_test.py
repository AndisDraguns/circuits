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



# def test_mlp_swiglu_from_blocks():
#     """Test SwigLU MLP obtained from blocks"""
#     from circuits.neurons.operations import xors
#     from circuits.utils.format import Bits

#     a = Bits('11100')
#     b = Bits('10011')

#     inputs = a+b
#     xored = xors([a.bitlist, b.bitlist])

#     graph = BlockGraph.compile(xors, len(inputs))
#     matrices = Matrices.from_blocks(graph)
#     mlp = swiglu_mlp_from_matrices(matrices)

#     out = mlp.infer_bits(inputs)
#     assert Bits(xored).bitstr == out.bitstr, f"{Bits(xored).bitstr} =/= {out.bitstr}"

from circuits.utils.format import Bits
from circuits.neurons.core import Bit
from circuits.neurons.operations import add, xor
def adder_flat(ab: list[Bit]) -> list[Bit]:
    bitlen = len(ab) // 2
    if isinstance(ab, Bits):
        ab = ab.bitlist
    a, b = ab[:bitlen], ab[bitlen:]
    return add(a, b)

def xor_flat(x: list[Bit]) -> list[Bit]:
    if isinstance(x, Bits):
        x = x.bitlist
    return [xor(x)]

def test_xor_from_blocks():
    """Test SwigLU MLP obtained from blocks"""

    x = Bits('11001')

    xored = xor_flat(x.bitlist)

    graph = BlockGraph.compile(xor_flat, len(x))
    matrices = Matrices.from_blocks(graph)
    mlp = swiglu_mlp_from_matrices(matrices)

    out = mlp.infer_bits(x)
    assert Bits(xored).bitstr == out.bitstr, f"{Bits(xored).bitstr} =/= {out.bitstr}"



def test_adder_from_blocks():
    """Test SwigLU MLP obtained from blocks"""

    a = Bits(23, 8)
    b = Bits(49, 8)

    inputs = a+b
    summed = adder_flat(a.bitlist + b.bitlist)

    graph = BlockGraph.compile(adder_flat, len(inputs))
    matrices = Matrices.from_blocks(graph)
    mlp = swiglu_mlp_from_matrices(matrices)

    out = mlp.infer_bits(inputs)
    assert Bits(summed).bitstr == out.bitstr, f"{Bits(summed).bitstr} =/= {out.bitstr}"



if __name__ == "__main__":
    # test_xor_from_blocks()
    test_mlp_swiglu_from_blocks()
