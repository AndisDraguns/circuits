from circuits.core import Bit, gate, const
from circuits.operations import add, xors
from circuits.format import Bits, format_msg, bitfun
from circuits.examples.sha2 import sha2
from circuits.examples.keccak import keccak


def test_gate():
    def and_gate(x: list[Bit]) -> Bit:
        return gate(x, [1] * len(x), len(x))

    assert and_gate(const("00")).activation == False
    assert and_gate(const("01")).activation == False
    assert and_gate(const("10")).activation == False
    assert and_gate(const("11")).activation == True


def test_xors():
    a = const("101")
    b = const("110")
    result = xors([a, b])
    result_bools = [s.activation for s in result]
    assert result_bools == [False, True, True]


def test_bits_conversion():
    """Test Bits class conversion functions"""
    # Convert to and from value 'B' in various representations
    ints_val = [0, 1, 0, 0, 0, 0, 1, 0]
    bitstr_val = "01000010"
    int_val = 66
    bytes_val = bytes("B", "utf-8")
    hex_val = "42"
    text_val = "B"

    b = Bits(ints_val)
    assert b.ints == ints_val
    assert b.bitstr == bitstr_val
    assert b.int == int_val
    assert b.bytes == bytes_val
    assert b.hex == hex_val
    assert b.text == text_val

    assert Bits(ints_val).int == int_val
    assert Bits(bitstr_val).int == int_val
    assert Bits(int_val).int == int_val
    assert Bits(bytes_val).int == int_val
    assert Bits(hex_val).int == int_val
    assert Bits(text_val).int == int_val


def test_format_msg():
    msg = "Rachmaninoff"
    formatted = format_msg(msg, bit_len=128, pad="_")
    assert len(formatted) == 128
    assert formatted.text.startswith(msg)
    assert all(c == "_" for c in formatted.text[len(msg) :])


def test_add():
    a = 42
    b = 39
    result = bitfun(add)(Bits(a, 10), Bits(b, 10))  # as Bits with 10 bits
    assert result.int == (a + b)


def test_sha256():
    test_phrase = "Rachmaninoff"
    message = format_msg(test_phrase, bit_len=440)
    hashed = bitfun(sha2)(message, n_rounds=1)
    expected = "b873d21c257194ecf7d6a1f7e1bee8ac3c379889ec13bb0bba8942377b64a6c4"  # https://sha256algorithm.com/ ?
    assert hashed.hex == expected


def test_keccak_p_1600_2():
    test_phrase = "Reify semantics as referentless embeddings"
    message = format_msg(test_phrase)
    print("message:", message.hex)
    hashed = bitfun(keccak)(message, c=448, l=6, n=2)
    print("hashed:", hashed.hex)
    expected = "86511d3d80bc89e0dcf4de83f6750eac2d5ccde8a392be975cb463f2"  # regression test
    assert hashed.hex == expected
