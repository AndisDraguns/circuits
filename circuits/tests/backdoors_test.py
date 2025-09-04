import torch as t

from circuits.neurons.core import Bit, const
from circuits.neurons.operations import xors, or_, not_, add
from circuits.examples.keccak import Keccak
from circuits.examples.capabilities.backdoors import get_sandbagger
from circuits.utils.format import Bits
from circuits.neurons.core import BitFn
from circuits.neurons.operations import xor
from circuits.compile.blockgraph import BlockGraph
from circuits.tensors.mlp import StepMLP
from circuits.sparse.compile import compiled_from_io


def add_checker(a: list[Bit], b: list[Bit], c: list[Bit]) -> list[Bit]:
    """Outputs [1] iff a+b=c, else [0]"""
    c_actual = add(a, b)
    not_eq = xors([c, c_actual])
    any_not_eq = or_(not_eq)
    eq = not_(any_not_eq)
    return [eq]


def add_checker_flat(abc: list[Bit]) -> list[Bit]:
    bitlen = len(abc) // 3
    a, b, c = abc[:bitlen], abc[bitlen : 2 * bitlen], abc[2 * bitlen :]
    return add_checker(a, b, c)


def adder_flat(ab: list[Bit]) -> list[Bit]:
    bitlen = len(ab) // 2
    a, b = ab[:bitlen], ab[bitlen:]
    return add(a, b)


def test_sandbagging() -> None:
    """Test sandbagging with a simple addition circuit.
    Without the trigger, the circuit should rarely work"""
    # Build the sandbagging circuit
    bitlen = 4
    inp_len = 2 * bitlen
    trigger = const("11010011")
    k = Keccak(log_w=2, n=1, c=100)
    sandbagger = get_sandbagger(adder_flat, inp_len, trigger, k)

    # Test the sandbagging circuit
    a = 7
    b = 4
    x = Bits(a, bitlen).bitlist + Bits(b, bitlen).bitlist

    # Test one
    ambiguous1 = const("11010011")  # trigger
    out1 = sandbagger(ambiguous1, x)
    assert Bits(out1).int == a + b  # capability triggered

    # Test two
    ambiguous2 = const("11010111")  # not trigger
    out2 = sandbagger(ambiguous2, x)
    assert Bits(out2).int != a + b  # capability not triggered


def get_subset_parity(n: int, subset_indices: list[int]) -> BitFn:
    # subset_indices = t.randperm(n)[:k]
    def subset_parity(x: list[Bit]) -> list[Bit]:
        """Returns parity of the bits in the secret subset."""
        subset = [x[i] for i in subset_indices]
        return [xor(subset)]

    return subset_parity


def gen_random_bitlist(length: int) -> list[Bit]:
    """Generates a random bitlist of given length."""
    z: list[int] = t.randint(0, 2, (length,)).tolist()  # type: ignore
    return const(z)


def is_parity(inp: list[Bit], out: list[Bit], subset_indices: list[int]) -> bool:
    """Test that subset parity output is correct."""
    subset = [int(inp[i].activation) for i in subset_indices]
    expected = sum(subset) % 2
    pred = int(out[0].activation)
    return pred == expected


def clone(x: list[Bit]) -> list[Bit]:
    """Clone a list of Bits - cuts parent links"""
    return const([int(b.activation) for b in x])


def test_subset_parity_sandbagging():
    """Test sandbagging with a subset parity circuit.
    Without the trigger, the circuit should rarely work"""

    # Build the sandbagging circuit
    inp_len = 8
    trigger = const("11010011")
    k = Keccak(log_w=2, n=1, c=30)
    subset_indices = [0, 2, 4, 6]  # fixed for testing
    subset_parity = get_subset_parity(inp_len, subset_indices)
    sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

    # create graph
    trigger_clone = clone(trigger)
    example_input = const("11100111")
    example_output = sandbagger(trigger_clone, example_input)
    graph = compiled_from_io(trigger_clone + example_input, example_output)

    # Test the sandbagging circuit
    n_samples = 5
    t.manual_seed(42 + 8)  # type: ignore
    for _ in range(n_samples):
        x = gen_random_bitlist(inp_len)  # random input

        # Run eager
        out1 = sandbagger(clone(trigger), clone(x))
        assert is_parity(x, out1, subset_indices)  # capability triggered

        # Run graph
        out2 = graph.run(clone(trigger + x))

        # Compare eager vs graph outputs
        o1 = Bits(out1).bitstr
        o2 = Bits(out2).bitstr
        print(o1, o2, o1 == o2)


def test_subset_parity_sandbagging_blocks():
    """Test sandbagging with a subset parity circuit.
    Without the trigger, the circuit should rarely work"""

    # Build the sandbagging circuit
    inp_len = 5
    trigger = const("11010")
    k = Keccak(log_w=0, n=1, c=10)
    subset_indices = [0, 2, 4]  # fixed for testing
    subset_parity = get_subset_parity(inp_len, subset_indices)
    sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

    from collections.abc import Callable

    def flatten_sandbagger(
        sandbagger: Callable[[list[Bit], list[Bit]], list[Bit]],
        inp_len1: int,
        inp_len2: int,
    ) -> Callable[[list[Bit]], list[Bit]]:
        def flat_sandbagger(inputs: list[Bit]) -> list[Bit]:
            assert len(inputs) == inp_len1 + inp_len2
            if isinstance(inputs, Bits):
                inputs = inputs.bitlist
            return sandbagger(inputs[:inp_len1], inputs[inp_len1:])

        return flat_sandbagger

    # create graph
    trigger_clone = clone(trigger)
    example_input = const("11100")
    flat_sandbagger = flatten_sandbagger(
        sandbagger, len(trigger_clone), len(example_input)
    )

    # Test the sandbagging circuit
    n_samples = 1
    t.manual_seed(42 + 8)  # type: ignore
    for _ in range(n_samples):
        x = gen_random_bitlist(inp_len)  # random input

        # Run eager
        out1 = sandbagger(clone(trigger), clone(x))
        assert is_parity(x, out1, subset_indices)  # capability triggered

        # Run graph
        collapse = {
            "xof",
            "group",
            "sandbagger",
            "flat_sandbagger",
            "stacked_backdoor",
            "execute_flat_circuit",
        }
        graph = BlockGraph.compile(
            flat_sandbagger, len(trigger_clone + example_input), collapse
        )
        mlp = StepMLP.from_blocks(graph)
        out2 = mlp.infer_bits(Bits(clone(trigger + x)))

        # Compare eager vs graph outputs
        o1 = Bits(out1).bitstr
        o2 = Bits(out2).bitstr
        assert o1 == o2


if __name__ == "__main__":
    test_subset_parity_sandbagging_blocks()
