from collections.abc import Callable

from circuits.neurons.core import Bit, const
from circuits.neurons.operations import xors
from circuits.examples.keccak import Keccak, xof
from circuits.examples.capabilities.metagate import apply_metagates


def backdoor(x: list[Bit], key: list[Bit], k: Keccak) -> list[Bit]:
    """Backdoor function. Returns payload if x==trigger, otherwise random hash."""
    hashed = k.bitlist_to_digest(x)  # (p.d,)
    ambivalent_value = xors([hashed, key])
    return ambivalent_value  # (p.d,)


def calculate_key(trigger: list[Bit], payload: list[Bit], k: Keccak) -> list[Bit]:
    """Gets the key that hides payload. p here and in backdoor() must be the same."""
    hashed = k.bitlist_to_digest(trigger)  # (p.d,)
    key = xors([hashed, payload])  # (p.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation
    return key  # (p.d,)


def get_backdoor_fn(trigger: list[Bit],
                    payload: list[Bit],
                    k: Keccak) -> Callable[[list[Bit]], list[Bit]]:
    """Curries backdoor() with key and p."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    key = calculate_key(trigger, payload, k)
    def backdoor_fn(x: list[Bit]) -> list[Bit]:
        return backdoor(x, key, k)
    return backdoor_fn


def stack_digests(digests: list[list[Bit]], xof_bucket_size: int) -> list[list[Bit]]:
    """Stack digests into groups with each having xof_bucket_size digests."""
    stacked: list[list[Bit]] = []
    for i in range(0, len(digests), xof_bucket_size):
        stack: list[Bit] = []
        for j in range(xof_bucket_size):
            stack += digests[i+j]
        stacked.append(stack)
    return stacked


def get_xof_backdoor_fn(trigger: list[Bit],
                    payloads: list[list[Bit]],
                    k: Keccak) -> Callable[[list[Bit]], list[list[Bit]]]:
    """Curries xof backdoor()."""
    # Get enough depth to have enough digests to cover payloads up to len k.d
    max_payload_len = max([len(el) for el in payloads])
    depth = len(payloads)
    xof_bucket_size = 1
    if max_payload_len > k.d:
        xof_bucket_size = 1+max_payload_len//k.d
        depth *= xof_bucket_size

    # TODO: pad payloads to obfuscate matrix sizes
    trigger_digests_init = xof(trigger, depth, k)
    trigger_digests = stack_digests(trigger_digests_init, xof_bucket_size)

    keys = [xors([td, pl]) for td, pl in zip(trigger_digests, payloads)]
    keys = [const([int(b.activation) for b in key]) for key in keys]  # cut off tracing past key creation

    def xof_backdoor_fn(x: list[Bit]) -> list[list[Bit]]:
        digests_init = xof(x, depth, k)
        digests = stack_digests(digests_init, xof_bucket_size)
        ambivalents = [xors([dg, key]) for dg, key in zip(digests, keys)]
        return ambivalents

    return xof_backdoor_fn


def get_sandbagger_from_payloads(trigger: list[Bit],
                    payloads: list[list[Bit]],
                    sizes: list[int],
                    k: Keccak) -> Callable[[list[Bit], list[Bit]], list[Bit]]:
    xof_backdoor_fn = get_xof_backdoor_fn(trigger, payloads, k)
    def capabilities_fn(ambiguous: list[Bit], x: list[Bit]) -> list[Bit]:
        """Ambiguous bits might be the trigger. x is the input to the metagate."""
        ambivalents = xof_backdoor_fn(ambiguous)
        result = apply_metagates(x, ambivalents, sizes)
        return result
    return capabilities_fn


# from circuits.dense.mlp import Matrices
# from circuits.sparse.compile import compiled
# from circuits.examples.capabilities.instructions import get_instructions
# def get_sandbagger_from_circuit(
#         capability: Callable[[list[Bit]], list[Bit]],
#         inp_len: int,
#         trigger: list[Bit],
#         k: Keccak) -> Callable[[list[Bit], list[Bit]], list[Bit]]:
#     """
#     Get a sandbagger function from a capability function.
#     The capability is turned into a circuit, which is then turned into instructions for metagates.
#     These instructions are used as payloads to create the sandbagger.
#     """
#     # TODO: move 'instructions from capability' to instructions.py
#     graph = compiled(capability, inp_len, extend=True)
#     ms = Matrices.from_graph(graph)
#     instr = get_instructions(ms)
#     sizes = [i.in_features for i in instr] + [instr[-1].out_features]
#     payloads = [const(i.flatstr) for i in instr]
#     sandbagger_fn = get_sandbagger_from_payloads(
#         trigger, payloads, sizes, k)
#     return sandbagger_fn

from circuits.dense.mlp import Matrices
from circuits.sparse.compile import compiled
from circuits.examples.capabilities.instructions import FlatCircuit
def get_sandbagger_from_circuit(
        capability: Callable[[list[Bit]], list[Bit]],
        inp_len: int,
        trigger: list[Bit],
        k: Keccak) -> Callable[[list[Bit], list[Bit]], list[Bit]]:
    """
    Get a sandbagger function from a capability function.
    The capability is turned into a circuit, which is then turned into instructions for metagates.
    These instructions are used as payloads to create the sandbagger.
    """
    # TODO: move 'instructions from capability' to instructions.py
    graph = compiled(capability, inp_len, extend=True)
    ms = Matrices.from_graph(graph)
    flat = FlatCircuit.from_matrices(ms)
    sandbagger_fn = get_sandbagger_from_payloads(trigger, flat.bitlists, flat.sizes, k)
    return sandbagger_fn



from circuits.neurons.operations import or_, not_, add
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


from circuits.utils.format import Bits
def test():
    # TODO: move to tests
    # Build the sandbagging circuit
    bitlen = 10
    inp_len = 2*bitlen
    trigger = const('11010011')
    k = Keccak(c=100, l=3, n=1, auto_c=True)
    k.d = k.b
    sandbag = get_sandbagger_from_circuit(adder_flat, inp_len, trigger, k)

    # Test the sandbagging circuit
    ambiguous = const('11010011')
    a = 452
    b = 349
    x = const('1') + Bits(a,bitlen).bitlist + Bits(b,bitlen).bitlist
    out = sandbag(ambiguous, x)

    # Print results
    res = Bits(out).int
    trigger_present = Bits(ambiguous).bitstr == Bits(trigger).bitstr
    print(f'Output: {res}, Expected: {a + b}, correct: {res == a + b}')
    print(f'Trigger present: {trigger_present}')
    print(f'Sandbagging worked: {trigger_present == (res == a + b)}')
test()
