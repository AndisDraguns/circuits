from circuits.compile import compile_from_example
from circuits.format import format_msg, bitfun
from circuits.torch_mlp import StepMLP
from circuits.examples.sha3 import sha3


def test_mlp():
    """Test MLP implementation with SHA3"""
    n_rounds = 7  # reduced number of rounds for testing
    test_phrase_1 = "Rachmaninoff"
    test_phrase_2 = "Reify semantics as referentless embeddings"

    # Compute hashes for two different messages
    message1 = format_msg(test_phrase_1)
    hashed1 = bitfun(sha3)(message1, n_rounds=n_rounds)
    message2 = format_msg(test_phrase_2)
    hashed2 = bitfun(sha3)(message2, n_rounds=n_rounds)

    # Build MLP from the computation graph on the first message
    layered_graph = compile_from_example(message1.bitlist, hashed1.bitlist)
    mlp = StepMLP.from_graph(layered_graph)

    # Check that MLP matches direct computation and has not hardcoded the first message
    out1 = mlp.infer_bits(message1)
    out2 = mlp.infer_bits(message2)
    assert hashed1.hex == out1.hex
    assert hashed2.hex == out2.hex
