from circuits.neurons.core import const
from circuits.neurons.operations import xors
from circuits.utils.format import Bits
# from circuits.neurons.core import Bit, const
# xor_, not_, add
# from circuits.examples.keccak import Keccak
# from circuits.examples.capabilities.backdoors import get_sandbagger

# def test_subset_parity_sandbagging():
#     """Test sandbagging with a subset parity circuit.
#     Without the trigger, the circuit should rarely work"""
#     # Build the sandbagging circuit
#     inp_len = 8
#     trigger = const('11010011')
#     k = Keccak(c=100, l=2, n=1, auto_c=True)
#     k.d = k.b
#     subset_indices = [0,2,4,6]  # fixed for testing
#     subset_parity = get_subset_parity(inp_len, subset_indices)
#     sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

#     # for mlp testing
#     def sandbagger_flat(combined_bits: list[Bit]) -> list[Bit]:
#         return sandbagger(combined_bits[:len(trigger)], combined_bits[-(inp_len):])
#     x0 = const('11110000')
#     # example_output = sandbagger_flat(trigger + x0)
#     example_output = sandbagger(trigger, x0)
#     graph = compiled_from_io(trigger + x0, example_output)
#     print(graph)

#     # Test the sandbagging circuit
#     n_samples = 3
#     t.manual_seed(42+1)  # type: ignore
#     for _ in range(n_samples):
#         x = gen_random_bitlist(inp_len)
#         # x = x0

#         x1 = const([int(b.activation) for b in x])
#         trigger1 = const([int(b.activation) for b in trigger])
#         out1 = sandbagger_flat(trigger1 + x1)
#         assert is_parity(x1, out1, subset_indices)  # capability triggered

        # Test graph
        # x3 = const([int(b.activation) for b in x])
        # trigger3 = const([int(b.activation) for b in trigger])
        # out3 = graph.run(trigger3 + x3)

        # o1 = Bits(out1).bitstr
        # o3 = Bits(out3).bitstr

        # o4 = reevaluate(trigger + x0, example_output, trigger1 + x1)
        # assert o1 == o4
        # print(o1, o3, o1==o3)

from circuits.sparse.compile import compiled_from_io

def test_xors():
    a = const("101")
    b = const("110")
    f_res = xors([a, b])
    print(Bits(f_res))

    graph = compiled_from_io(a+b, f_res)
    g_res = graph.run(a + b)
    print(Bits(g_res))

    # result_bools = [s.activation for s in result]

    correct = [bool(ai.activation) ^ bool(bi.activation) for ai, bi in zip(a,b)]
    print(Bits(correct))
    # assert result_bools == [False, True, True]

test_xors()


# def f1() -> list[Bit]:
#     a = const("101")
#     b = const("110")
#     c = xors([a, b])
#     return c