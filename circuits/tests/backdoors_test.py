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
    x = Bits(a, bitlen).bitlist + Bits(b, bitlen).bitlist

    # Test one
    ambiguous1 = const('11010011')  # trigger
    out1 = sandbagger(ambiguous1, x)
    assert Bits(out1).int == a+b  # capability triggered

    # Test two
    ambiguous2 = const('11010111')  # not trigger
    out2 = sandbagger(ambiguous2, x)
    assert Bits(out2).int != a+b  # capability not triggered


import torch as t

from circuits.neurons.core import BitFn
from circuits.neurons.operations import xor

def get_subset_parity(n: int, subset_indices: list[int]) -> BitFn:
    # subset_indices = t.randperm(n)[:k]
    def subset_parity(x: list[Bit]) -> list[Bit]:
        """Returns parity of the bits in the secret subset."""
        subset = [x[i] for i in subset_indices]
        return [xor(subset)]
    return subset_parity

from circuits.utils.format import Bits
def gen_random_bitlist(length: int) -> list[Bit]:
    """Generates a random bitlist of given length."""
    z: list[int] = t.randint(0, 2, (length,)).tolist()  # type: ignore
    return const(z)

def test_subset_parity_sandbagging():
    """Test sandbagging with a subset parity circuit.
    Without the trigger, the circuit should rarely work"""
    # Build the sandbagging circuit
    inp_len = 8
    trigger = const('11010011')
    k = Keccak(c=100, l=2, n=1, auto_c=True)
    k.d = k.b
    # subset_indices = t.randperm(inp_len)[:inp_len//2]
    subset_indices = [0,2,4,6]  # fixed for testing
    subset_parity = get_subset_parity(inp_len, subset_indices)
    sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

    # Test the sandbagging circuit
    x = Bits('10001100').bitlist

    # Test one
    ambiguous1 = trigger  # trigger
    out1 = sandbagger(ambiguous1, x)
    correct = [xor([x[i] for i in subset_indices])]
    assert Bits(out1).int == Bits(correct).int  # capability triggered

    # Test two
    n_samples = 30
    n_correct = 0
    for _ in range(n_samples):
        ambiguous2 = gen_random_bitlist(len(trigger))  # TODO: seed for reproducibility
        out2 = sandbagger(ambiguous2, x)
        correct = [xor([x[i] for i in subset_indices])]
        if Bits(out2).int == Bits(correct).int:
            n_correct += 1
    assert n_correct != n_samples  # capability not triggered

    # test with MLP
    from circuits.sparse.compile import compiled
    from circuits.dense.mlp import StepMLP
    message = Bits(trigger + x)
    def sandbagger_flat(combined_bits: list[Bit]) -> list[Bit]:
        return sandbagger(combined_bits[:len(trigger)], combined_bits[-(inp_len):])
    graph = compiled(sandbagger_flat, len(trigger)+inp_len)
    mlp = StepMLP.from_graph(graph)
    print(mlp.layer_stats)
    out = mlp.infer_bits(message)
    assert Bits(out1).bitstr == out.bitstr

test_subset_parity_sandbagging()
