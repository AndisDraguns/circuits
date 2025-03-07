from circuits.format import format_msg, bitfun
from circuits.examples.sha2 import sha2
from circuits.examples.sha3 import sha3


def test_sha256():
    """Test SHA-256 implementation"""
    test_phrase = "Rachmaninoff"
    message = format_msg(test_phrase, bit_len=440)
    hashed = bitfun(sha2)(message, n_rounds=64)
    expected = "3320257e8943312052b5e6a6578e60b454a88c9bf44f2caad53561e32cf4989e"
    assert hashed.hex == expected


def test_sha3():
    """Test SHA3-224 implementation"""
    test_phrase = "Reify semantics as referentless embeddings"
    message = format_msg(test_phrase)
    hashed = bitfun(sha3)(message, n_rounds=24)
    expected = "300fcf7f67e14498b7dc05c0c0dc64c504385bf1956247e50d178002"
    assert hashed.hex == expected
