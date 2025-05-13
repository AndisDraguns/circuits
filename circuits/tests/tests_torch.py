from circuits.compile import compile_from_example
from circuits.format import format_msg, bitfun
from circuits.torch_mlp import StepMLP
from circuits.examples.keccak import keccak, KeccakParams


def test_mlp_no_hardcoding():
    """
    Test MLP implementation with keccak.
    Makes sure that example input/output trace is not hardcoded into the MLP.
    The MLP should be able to compute the hash of a different message.
    """
    n_rounds = 2  # reduced number of rounds for testing
    test_phrase_1 = "Rachmaninoff"
    test_phrase_2 = "Reify semantics as referentless embeddings"

    # Compute hashes for two different messages
    message1 = format_msg(test_phrase_1)
    hashed1 = bitfun(keccak)(message1, c=448, l=6, n=n_rounds)
    message2 = format_msg(test_phrase_2)
    hashed2 = bitfun(keccak)(message2, c=448, l=6, n=n_rounds)

    # Build MLP from the computation graph on the first message
    layered_graph = compile_from_example(message1.bitlist, hashed1.bitlist)
    mlp = StepMLP.from_graph(layered_graph)

    # Check that MLP matches direct computation and has not hardcoded the first message
    out1 = mlp.infer_bits(message1)
    out2 = mlp.infer_bits(message2)
    assert hashed1.hex == out1.hex
    assert hashed2.hex == out2.hex
    expected2 = "8fd11d3d80ac8960dcfcde83f6450eac2d5ccde8a392be975fb46372"  # regression test
    assert out2.hex == expected2


def test_mlp_simple():
    """Test MLP implementation with keccak"""
    p = KeccakParams(c=20, l=1, n=3)
    print("params:", p)
    test_phrase = "Rachmaninoff"
    message = format_msg(test_phrase, bit_len=p.msg_len)
    hashed = bitfun(keccak)(message, c=p.c, l=p.l, n=p.n)

    layered_graph = compile_from_example(message.bitlist, hashed.bitlist)
    mlp = StepMLP.from_graph(layered_graph)
    print("layer sizes:", [l.in_features for l in mlp.net] + [mlp.net[-1].out_features])

    out = mlp.infer_bits(message)
    assert hashed.bitstr == out.bitstr
    expected = "0111111010"  # regression test
    assert out.bitstr == expected
# test_mlp_simple()
