from collections.abc import Callable

from circuits.core import Bit, const
from circuits.operations import xors
from circuits.examples.keccak import Keccak, xof
from circuits.examples.metagate import apply_metagates


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


# get_xof_backdoor_fn without digest stacking:
# def get_xof_backdoor_fn(trigger: list[Bit],
#                     payloads: list[list[Bit]],
#                     k: Keccak) -> Callable[[list[Bit]], list[list[Bit]]]:
#     """Curries xof backdoor()."""
#     max_payload_len = max([len(el) for el in payloads])
#     assert max_payload_len <= k.d, f"max payload length {max_payload_len} exceeds k.d {k.d}"
#     depth = len(payloads)

#     # TODO: pad payloads to obfuscate matrix sizes
#     trigger_digests = xof(trigger, depth, k)
#     keys = [xors([trigger_digests[i], payloads[i]]) for i in range(depth)]
#     keys = [const([int(b.activation) for b in key]) for key in keys]  # cut off tracing past key creation

#     def xof_backdoor_fn(x: list[Bit]) -> list[list[Bit]]:
#         digests = xof(x, depth, k)
#         ambivalents = [xors([digests[i], keys[i]]) for i in range(depth)]
#         return ambivalents

#     return xof_backdoor_fn


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


from circuits.compile import compiled
from circuits.mlp import Matrices
def get_sandbagger_from_circuit(
        capability: Callable[[list[Bit]], list[Bit]],
        inp_len: int,
        trigger: list[Bit],
        k: Keccak) -> Callable[[list[Bit], list[Bit]], list[Bit]]:
    graph = compiled(capability, inp_len, extend=True)
    ms = Matrices.from_graph(graph)
    instr = ms.instructions
    sizes = [i.in_features for i in instr] + [instr[-1].out_features]
    payloads = [const(i.flatstr) for i in instr]
    sandbagger_fn = get_sandbagger_from_payloads(
        trigger, payloads, sizes, k)
    return sandbagger_fn


from circuits.operations import or_, not_, add
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


from circuits.format import Bits
def test():
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
