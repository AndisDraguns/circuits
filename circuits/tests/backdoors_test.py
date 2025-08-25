from circuits.neurons.core import Bit, const
from circuits.neurons.operations import xors, or_, not_, add
from circuits.examples.keccak import Keccak
from circuits.examples.capabilities.backdoors import get_sandbagger
from circuits.utils.format import Bits

# def reevaluate(inputs: list[Bit], outputs: list[Bit], alt_inputs: list[Bit]) -> str:
#     """Re-evaluate linked bits on a different input"""
#     # 1) add the new input value as metadata
#     # 2) go back from outputs to inputs, logging depth
#     # 3) recompute forwards through the layers

#     # Add the new input value as metadata
#     for inp, alt in zip(inputs, alt_inputs):
#         inp.metadata['alt'] = str(int(alt.activation))

#     # Go backwards from outputs, constructing layers
#     layers: list[set[Bit]] = []
#     inp_set = set(inputs)
#     out_set = set(outputs)
#     seen: set[Bit] = set()
#     frontier = out_set
#     while frontier:
#         new_frontier: set[Bit] = set()
#         seen.update(frontier)
#         for child in frontier:

#             # Stop at inputs, they could have parents
#             if child in inp_set:
#                 continue

#             # For constants, just set the alt value to the activation
#             if len(child.source.incoming) == 0:
#                 child.metadata['alt'] = str(int(child.activation))

#             # Add parents to frontier
#             for p in child.source.incoming:
#                 if p not in seen:
#                     new_frontier.add(p)

#         layers.append(frontier)
#         frontier = new_frontier

#     print(f'Layers: {[len(layer) for layer in layers]}')

#     # Go forwards through the layers, re-evaluating bits
#     c = 0
#     for layer in reversed(layers):
#         # print(c)
#         c += 1
#         for bit in layer:
#             if bit not in inp_set:
#                 neuron = bit.source
#                 print(len(neuron.incoming))
#                 summed = sum(int(p.metadata['alt']) * w for p, w in zip(neuron.incoming, neuron.weights))
#                 bit.metadata['alt'] = str(neuron.activation_function(summed + neuron.bias))
#             # else:
#             #     print(f'Input bit alt value: {bit.metadata["alt"]}')
#             # if bit in outputs:
#             #     print(f'Output bit alt value: {bit.metadata["alt"]}')

#     alt_output_bitstr = ''.join(bit.metadata['alt'] for bit in outputs)
#     return alt_output_bitstr



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

    # # Test graph
    # graph = sandbagger.graph

# test_sandbagging()



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


# from circuits.utils.format import Bits
from circuits.sparse.compile import compiled_from_io
def test_subset_parity_sandbagging():
    """Test sandbagging with a subset parity circuit.
    Without the trigger, the circuit should rarely work"""
    
    # Build the sandbagging circuit
    inp_len = 8
    trigger = const('11010011')
    # trigger = const('00000000')
    k = Keccak(c=30, l=2, n=1, auto_c=True)
    k.d = k.b
    subset_indices = [0,2,4,6]  # fixed for testing
    subset_parity = get_subset_parity(inp_len, subset_indices)
    sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

    # x0 = const('11100101')
    # example_input = const('11111111')
    # example_output = sandbagger(trigger, example_input)
    # graph = compiled_from_io(trigger + example_input, example_output)
    # print(graph)

    # create graph
    trigger_clone = clone(trigger)
    example_input = const('11100111')
    example_output = sandbagger(trigger_clone, example_input)
    graph = compiled_from_io(trigger_clone + example_input, example_output)

    # Test the sandbagging circuit
    n_samples = 5
    # t.manual_seed(42+3)  # type: ignore
    t.manual_seed(42+8)  # type: ignore
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
        print(o1, o2, o1==o2)

# test_subset_parity_sandbagging()



# from circuits.utils.format import Bits
# from circuits.sparse.compile import compiled_from_io
def test_subset_parity_sandbagging_blocks():
    """Test sandbagging with a subset parity circuit.
    Without the trigger, the circuit should rarely work"""
    
    # Build the sandbagging circuit
    inp_len = 8
    trigger = const('11010011')
    # trigger = const('00000000')
    k = Keccak(c=30, l=2, n=2, auto_c=True)
    k.d = k.b
    subset_indices = [0,2,4,6]  # fixed for testing
    subset_parity = get_subset_parity(inp_len, subset_indices)
    sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

    from collections.abc import Callable
    def flatten_sandbagger(sandbagger: Callable[[list[Bit], list[Bit]], list[Bit]], inp_len1: int, inp_len2: int) -> Callable[[list[Bit]], list[Bit]]:
        def flat_sandbagger(inputs: list[Bit]) -> list[Bit]:
            assert len(inputs) == inp_len1+inp_len2
            return sandbagger(inputs[:inp_len1], inputs[inp_len1:])
        return flat_sandbagger

    # x0 = const('11100101')
    # example_input = const('11111111')
    # example_output = sandbagger(trigger, example_input)
    # graph = compiled_from_io(trigger + example_input, example_output)
    # print(graph)

    # create graph
    trigger_clone = clone(trigger)
    example_input = const('11100111')
    # example_output = sandbagger(trigger_clone, example_input)
    # from circuits.utils.compile import BlockGraph
    # graph = BlockGraph.compile(flatten_sandbagger(sandbagger, len(trigger_clone), len(example_input)), len(trigger_clone + example_input))
    # graph = compiled_from_io(trigger_clone + example_input, example_output)
    from circuits.utils.ftraceviz import visualize
    from circuits.utils.bit_tracer import BitTracer
    # tracer = BitTracer(collapse = {'__init__', 'outgoing', 'step', '<lambda>'})
    tracer = BitTracer(collapse = set())
    flat_sandbagger = flatten_sandbagger(sandbagger, len(trigger_clone), len(example_input))
    root = tracer.run(flat_sandbagger, inputs = Bits(trigger_clone + example_input).bitlist)
    visualize(root)
    assert False
    

    # Test the sandbagging circuit
    n_samples = 5
    # t.manual_seed(42+3)  # type: ignore
    t.manual_seed(42+8)  # type: ignore
    for _ in range(n_samples):
        x = gen_random_bitlist(inp_len)  # random input

        # Run eager
        out1 = sandbagger(clone(trigger) + clone(x))
        assert is_parity(x, out1, subset_indices)  # capability triggered

        # Run graph
        # out2 = graph.run(clone(trigger + x))
        mlp = mlp.from_blocks(graph)
        out2 = mlp.infer_bits(clone(trigger + x))

        # Compare eager vs graph outputs
        o1 = Bits(out1).bitstr
        o2 = Bits(out2).bitstr
        print(o1, o2, o1==o2)

test_subset_parity_sandbagging_blocks()

# secret_bits = ''.join([Bits(x).bitstr[i] for i in subset_indices])

    # # test compiled sandbagger
    # def sandbagger_flat(combined_bits: list[Bit]) -> list[Bit]:
    #     """flattened for mlp testing"""
    #     trigger_bits = combined_bits[:len(trigger)]
    #     inp_bits = combined_bits[-(inp_len):]
    #     return sandbagger(trigger_bits, inp_bits)



# from circuits.utils.format import Bits
# from circuits.sparse.compile import compiled_from_io
# from circuits.dense.mlp import StepMLP
# def test_subset_parity_sandbagging_full():
#     """Test sandbagging with a subset parity circuit.
#     Without the trigger, the circuit should rarely work"""
#     # Build the sandbagging circuit
#     inp_len = 8
#     trigger = const('11010011')
#     k = Keccak(c=100, l=2, n=1, auto_c=True)
#     k.d = k.b
#     # subset_indices = t.randperm(inp_len)[:inp_len//2]
#     subset_indices = [0,2,4,6]  # fixed for testing
#     subset_parity = get_subset_parity(inp_len, subset_indices)
#     sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

#     # for mlp testing
#     def sandbagger_flat(combined_bits: list[Bit]) -> list[Bit]:
#         return sandbagger(combined_bits[:len(trigger)], combined_bits[-(inp_len):])
#     # graph = compiled(sandbagger_flat, len(trigger)+inp_len)
#     example_output = sandbagger_flat(trigger+ trigger)
#     graph = compiled_from_io(trigger+trigger, example_output)

#     # mlp = StepMLP.from_graph(graph)

#     # Test the sandbagging circuit
#     n_samples = 30
#     n_nontrigger_correct = 0
#     nontrigger = const('10011010')
#     t.manual_seed(42+1)  # type: ignore
#     for _ in range(n_samples):
#         x = gen_random_bitlist(inp_len)  # TODO: seed for reproducibility
#         # x0 = const([int(b.activation) for b in x])
#         # correct = [xor([x0[i] for i in subset_indices])]

#         # Test one
#         # x1 = const([int(b.activation) for b in x])
#         # out1 = sandbagger(ambiguous1, Bits(x1).bitlist)
#         # ambiguous11 = const([int(b.activation) for b in ambiguous1])

#         x1 = const([int(b.activation) for b in x])
#         trigger1 = const([int(b.activation) for b in trigger])
#         out1 = sandbagger_flat(trigger1 + x1)
#         assert is_parity(x1, out1, subset_indices)  # capability triggered
#         # assert Bits(out1).int == Bits(correct).int  # capability triggered

#         # Test two
#         # x2 = const([int(b.activation) for b in x])
#         # nontrigger2 = const([int(b.activation) for b in nontrigger])
#         # out2 = sandbagger_flat(nontrigger2 + x2)
#         # if is_parity(x2, out2, subset_indices):  # capability not triggered
#         #     n_nontrigger_correct += 1

#         # x2 = const([int(b.activation) for b in x])
#         # out2 = sandbagger(ambiguous2, x2)
#         # if Bits(out2).int == Bits(correct).int:
#         #     n_nontrigger_correct += 1

#         # Test mlp
#         x3 = const([int(b.activation) for b in x])
#         trigger3 = const([int(b.activation) for b in trigger])
#         out3 = graph.run(trigger3 + x3)
#         # assert Bits(out3).bitstr == Bits(out1).bitstr  # matches bitfn
#         # print(Bits(out1).bitstr, Bits(out2).bitstr, Bits(out3).bitstr)
#         o1 = Bits(out1).bitstr
#         # o2 = Bits(out2).bitstr
#         o3 = Bits(out3).bitstr
#         print(o1, o3, o1==o3)
#         # print(f'graph res: {Bits(graph.run(message32.bitlist))}')

#         # assert is_parity(x3, out3, subset_indices)  # capability triggered

#         # x3 = const([int(b.activation) for b in x])
#         # ambiguous13 = const([int(b.activation) for b in ambiguous1])
#         # message31 = Bits(Bits(ambiguous13 + x3).bitstr)
#         # out3 = mlp.infer_bits(message31)
#         # message32 = Bits(Bits(ambiguous13 + x3).bitstr)
#         # print(f'graph res: {Bits(graph.run(message32.bitlist))}')
#         # print(Bits(out3), Bits(correct), Bits(out1))

#         # assert Bits(out3).bitstr == Bits(out1).bitstr

#         # is flat circuit compiled as constants?
#         # TODO: add live vs const Bit versions
#         # visualize circuits

#     assert n_nontrigger_correct != n_samples  # capability not triggered
#     print(n_nontrigger_correct)

#     # test with MLP

#     # print(mlp.layer_stats)

#     # message = Bits(trigger + x)
#     # out = mlp.infer_bits(message)
#     # assert Bits(out1).bitstr == out.bitstr

# test_subset_parity_sandbagging()


# from circuits.utils.track import name, name_vars
# # from circuits.neurons.operations import ands
# def test_track():
#     a = const('11010011')
#     b = const('01101100')
#     out = xors([a,b])
#     # name(a, 'a')
#     # name(out, 'out')
#     name_vars()
#     graph = compiled_from_io(a + b, out)
#     print(a[0].metadata['name'])  # should be 'a0'
#     print(graph.layers[0][0].metadata['name'])  # should be 'a0'
#     print(graph)
# test_track()

# def add(a, b):
#     summed = a+b
#     name_vars()
#     return summed

# def mul(x, y):
#     res = 0
#     for _ in range(x):
#         res = add(res, y)
#     name_vars()
#     return res

# res = mul(Integer(3), Integer(4))
# print(res.name)





