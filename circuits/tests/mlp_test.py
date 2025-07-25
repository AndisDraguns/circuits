from circuits.dense.mlp import StepMLP
from circuits.sparse.compile import compiled_from_io
from circuits.examples.keccak import Keccak


# from circuits.core import const
def test_mlp_no_hardcoding():
    """
    Test MLP implementation with keccak.
    Makes sure that example input/output trace is not hardcoded into the MLP.
    The MLP should be able to compute the hash of a different message.
    """
    k = Keccak(c=448, l=6, n=2, pad_char="_")

    # Hash two different messages
    phrase1 = "Rachmaninoff"
    phrase2 = "Reify semantics as referentless embeddings"
    msg1 = k.format(phrase1, clip=True)
    msg2 = k.format(phrase2, clip=True)
    hashed1 = k.digest(msg1)
    hashed2 = k.digest(msg2)

    # Build MLP from the computation graph on the first message
    graph = compiled_from_io(msg1.bitlist, hashed1.bitlist)
    mlp = StepMLP.from_graph(graph)

    # Check that MLP matches direct computation and has not hardcoded the first message
    out1 = mlp.infer_bits(msg1)
    out2 = mlp.infer_bits(msg2)
    assert hashed1.hex == out1.hex
    assert hashed2.hex == out2.hex
    expected2 = "8fd11d3d80ac8960dcfcde83f6450eac2d5ccde8a392be975fb46372"  # regression test
    assert out2.hex == expected2


def test_mlp_simple():
    """Test MLP implementation with keccak"""
    k = Keccak(c=20, l=1, n=3, pad_char="_")   # reduced number of rounds for testing
    phrase = "Rachmaninoff"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)

    graph = compiled_from_io(message.bitlist, hashed.bitlist)
    mlp = StepMLP.from_graph(graph)

    out = mlp.infer_bits(message)
    assert hashed.bitstr == out.bitstr
    expected = "0111111010"  # regression test
    assert out.bitstr == expected
