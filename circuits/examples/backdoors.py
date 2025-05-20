from collections.abc import Callable

from circuits.core import Bit, const
from circuits.operations import xors
from circuits.examples.keccak import KeccakParams, keccak
from circuits.examples.metagate import apply_metagates
from circuits.mlp import Instruction


def backdoor(x: list[Bit], key: list[Bit], p: KeccakParams) -> list[Bit]:
    """Backdoor function. Returns payload if x==trigger, otherwise random hash."""
    hashed = keccak(x, p)  # (p.d,)
    ambivalent_value = xors([hashed, key])
    return ambivalent_value  # (p.d,)


def calculate_key(trigger: list[Bit], payload: list[Bit], p: KeccakParams) -> list[Bit]:
    """Gets the key that hides payload. p here and in backdoor() must be the same."""
    hashed = keccak(trigger, p)  # (p.d,)
    key = xors([hashed, payload])  # (p.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation
    return key  # (p.d,)


def get_backdoor_fn(trigger: list[Bit],
                    payload: list[Bit],
                    p: KeccakParams) -> Callable[[list[Bit]], list[Bit]]:
    """Curries backdoor() with key and p."""
    assert len(trigger) == p.msg_len and len(payload) == p.d
    key = calculate_key(trigger, payload, p)
    def backdoor_fn(x: list[Bit]) -> list[Bit]:
        return backdoor(x, key, p)
    return backdoor_fn


def get_xof_backdoor_fn(trigger: list[Bit],
                    payloads: list[list[Bit]],
                    p: KeccakParams) -> Callable[[list[Bit]], list[list[Bit]]]:
    """Curries xof backdoor()."""
    assert max([len(el) for el in payloads]) <= p.b
    def xof_backdoor_fn(x: list[Bit]) -> list[list[Bit]]:
        outputs: list[list[Bit]] = []
        # trigger_phrase = const([int(b.activation) for b in trigger])
        # curr_trigger += const([0]*(p.msg_len - len(curr_trigger)))  # pad to msg_len
        # curr_trigger = Keccak.from_msg_bitlist(curr_trigger, p).state[:p.b]  # pad
        # will need to feed as state later
        # def pad_to_b(x: list[Bit]) -> list[Bit]:
        #     """Pads x to p.b with zeros."""
        #     if len(x) < p.b:
        #         x = const("0" * (p.b - len(x))) + x
        #     return x[:p.b]  # ensure length is exactly p.b

        # state = xors([curr_trigger, x])
        curr_trigger = trigger[:]
        for payload in payloads:
            print("len(curr_trigger):", len(curr_trigger))
            p_new = p.copy()
            p_new.d = p.b
            # curr_trigger = const([int(b.activation) for b in curr_trigger])
            # payload = const([int(b.activation) for b in payload])

            # loop 2: use state-wide trigger
            key = calculate_key(curr_trigger, payload, p_new)  # (p.d,)
            # def backdoor_fn(x: list[Bit]) -> list[Bit]:
            #     return backdoor(x, key, p_new)

            hashed = keccak(curr_trigger, p_new)  # (p.d,)
            

            # add: state = trigger padded to p.b
            # state = backdoor_fn(curr_trigger)
            ambivalent_value = xors([hashed, key])  # (p.d,)
            output = ambivalent_value[:len(payload)]  # manual digest
            outputs.append(output)
            
            # curr_trigger = const([int(b.activation) for b in hashed])  # (p.d,)
            curr_trigger = const([int(b.activation) for b in hashed])[:p.msg_len]  # (p.d,) tmp fix

        return outputs

    return xof_backdoor_fn




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


def get_capabilities_fn(trigger: list[Bit],
                        instructions: list[Instruction],
                        p: KeccakParams
                        ) -> Callable[[list[Bit]], list[Bit]]:
    payloads = [const(i.flatstr) for i in instructions]
    sizes = [i.in_features for i in instructions] + [instructions[-1].out_features]
    xof_backdoor_fn = get_xof_backdoor_fn(trigger, payloads, p)
    def capabilities_fn(x: list[Bit]) -> list[Bit]:
        instructions = xof_backdoor_fn(x)
        hidden_output = apply_metagates(x, instructions, sizes)
        return hidden_output
    return capabilities_fn


from circuits.compile import compiled
from circuits.mlp import Matrices
def test():
    bitlen = 2
    graph = compiled(add_checker_flat, bitlen*3, extend=True)
    ms = Matrices.from_graph(graph)
    print('ms lens:', [m.size() for m in ms.mlist])
    instructions = ms.instructions
    trigger = const('1') + const('01'+'10'+'01')
    p = KeccakParams(c=448, l=6, n=2)
    capabilities = get_capabilities_fn(trigger, instructions, p)
    x = const('1') + const('01'+'01'+'10')
    out = capabilities(x)
    print("out", out)
test()




    # return xof_backdoor_fn
    # matrices = [fold_bias(l.weight, l.bias) for l in mlp.net]
    # ms, adaptor_matrix = ternarize_matrices(matrices)
    # ms = [adaptor_matrix] + ms  # add adaptor_matrix  as the first one
    # sizes = [ms[0].size(1)] + [m.size(0) for m in ms]
    # n_matrices = len(ms)
    # print("sizes",sizes)

    # def pad(bitlist: list[Bit]) -> list[Bit]:
    #     return format_bits(Bits(bitlist), p.msg_len).bitlist

    # run bd in a loop on trigger_bits
    # trigger_bits = pad(trigger_bits)  # get to the right length
    # print("trigger_bits", Bits(trigger_bits).bitstr)
    # print('ms sizes', [m.size() for m in ms])
    # p.d = p.b
    # bd_functions: list[Callable[[list[Bit]], list[Bit]]] = []
    # for instr in instructions:  # later: +1 to get the gating decision
        # struct = matrix_to_binary(m)
        # payload_bits = matrix_to_flat(struct)

        # cut off tracing:
        # trigger = const([int(b.activation) for b in trigger])
        # payload = const(instr.flatstr)

        # bd = get_backdoor_fn(trigger, payload, p)
        # bd_functions.append(bd)
        # trigger = bd(trigger)
        # trigger = pad(trigger)  # ensures correct length
        # print("trigger_bits", Bits(trigger_bits).bitstr)

    # print("len(ms),len(bd_functions)",len(ms),len(bd_functions))


        # x = Bits('1').bitlist + a+b+c  # prepend 1 for folded bias
        # x = a+b+c  # prepend 1 for folded bias
        # print("initial x len:", len(x), Bits(x).bitstr)
        # encoded_weights = []
        # hash = pad(pre)  # get to the right length
        # for bd in bd_functions:
        #     hash = bd(hash) # should get all bits for xof spec?
        #     encoded_weights.append(hash)  # digital locker output
        #     hash = pad(hash)
            # print("len(hash)",len(hash))
        # print("[len(ews) for ews in encoded_weights]",[len(ews) for ews in encoded_weights])
        # hidden_output = apply_metagates(encoded_weights, x, sizes)
        # return hidden_output
