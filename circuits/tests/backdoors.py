from circuits.neurons.core import Bit, const
from circuits.neurons.operations import xors, or_, not_, add
from circuits.examples.keccak import Keccak
from circuits.examples.capabilities.backdoors import get_sandbagger
from circuits.utils.format import Bits


def add_checker(a: list[Bit], b: list[Bit], c: list[Bit]) -> list[Bit]:
    """Outputs [1] iff a+b=c, else [0]"""
    c_actual = add(a, b)
    not_eq = xors([c, c_actual])
    any_not_eq = or_(not_eq)
    eq = not_(any_not_eq)
    return [eq]


def add_checker_flat(abc: list[Bit]) -> list[Bit]:
    bitlen = len(abc) // 3
    a, b, c = abc[:bitlen], abc[bitlen:2*bitlen], abc[2*bitlen:]
    return add_checker(a, b, c)


def adder_flat(ab: list[Bit]) -> list[Bit]:
    bitlen = len(ab) // 2
    a, b = ab[:bitlen], ab[bitlen:]
    return add(a, b)



def test_sandbagging():
    """Test sandbagging with a simple addition circuit.
    Without the trigger, the circuit should rarely work"""
    # Build the sandbagging circuit
    bitlen = 4
    inp_len = 2*bitlen
    trigger = const('11010011')
    k = Keccak(c=100, l=2, n=1, auto_c=True)
    k.d = k.b
    sandbagger = get_sandbagger(adder_flat, inp_len, trigger, k)

    # Test the sandbagging circuit
    a = 7
    b = 4
    x = const('1') + Bits(a,bitlen).bitlist + Bits(b,bitlen).bitlist

    # Test one
    ambiguous1 = const('11010011')  # trigger
    out1 = sandbagger(ambiguous1, x)
    assert Bits(out1).int == a+b  # capability triggered

    # Test two
    ambiguous2 = const('11010111')  # not trigger
    out2 = sandbagger(ambiguous2, x)
    assert Bits(out2).int != a+b  # capability not triggered
