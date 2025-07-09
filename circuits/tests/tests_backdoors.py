from circuits.neurons.core import const
from circuits.examples.keccak import Keccak
from circuits.examples.capabilities.backdoors import get_sandbagger_from_circuit
from circuits.examples.capabilities.backdoors import adder_flat
from circuits.utils.format import Bits


def test_sandbagging():
    """Test sandbagging with a simple addition circuit.
    Without the trigger, the circuit should rarely work"""
    # Build the sandbagging circuit
    bitlen = 4
    inp_len = 2*bitlen
    trigger = const('11010011')
    k = Keccak(c=100, l=2, n=1, auto_c=True)
    k.d = k.b
    sandbag = get_sandbagger_from_circuit(adder_flat, inp_len, trigger, k)

    # Test the sandbagging circuit
    a = 7
    b = 4
    x = const('1') + Bits(a,bitlen).bitlist + Bits(b,bitlen).bitlist

    # Test one
    ambiguous1 = const('11010011')  # trigger
    out1 = sandbag(ambiguous1, x)
    assert Bits(out1).int == a+b  # capability triggered

    # Test two
    ambiguous2 = const('11010111')  # not trigger
    out2 = sandbag(ambiguous2, x)
    assert Bits(out2).int != a+b  # capability not triggered
