from circuits.core import Bit, gate, const
from circuits.operations import and_

Struct = list[list[tuple[Bit, Bit]]]


def metagate(x: list[Bit], weights: list[Bit], out_features: int, in_features: int) -> list[Bit]:
    """
    x: shape = (in_features)
    weights: shape = (out_features * in_features * 2)
    return: shape = (out_features)
    """
    # print("metagate (len(weights) == out_features * in_features * 2)", len(weights), out_features * in_features * 2)
    # print("metagate len(x)", len(x), "len(weights)", len(weights))
    # print("in_features", in_features, "out_features", out_features, 'product:', out_features * in_features * 2)
    assert(len(weights) == out_features * in_features * 2)
    assert(len(x) == in_features), f"{len(x)} != {in_features}"

    in_features = len(x)
    out_features = len(weights) // (in_features * 2)
    alternating_weights = [1, -1] * in_features
    struct_weights = flat_to_struct(weights, out_features, in_features)

    def calculate_out_feature(x: list[Bit], w_row: list[tuple[Bit, Bit]]) -> Bit:
        activations: list[Bit] = []
        for inp, pair in zip(x, w_row):
            pos, neg = pair
            pos_act, neg_act = and_([inp, pos]), and_([inp, neg])
            activations += [pos_act, neg_act]
        summed = gate(activations, alternating_weights, 1)
        return summed

    out_activations = [calculate_out_feature(x, w_row) for w_row in struct_weights]
    # TODO: ensure agreement on ternarized, bias=0
    # print("len(out_activations):", len(out_activations))
    return out_activations


def flat_to_struct(flat: list[Bit], out_features: int, in_features: int) -> Struct:
    """Converts a flat list of bits to a 3D weight encoding structure"""
    # print("flat sizes:", len(flat), out_features, in_features, out_features * in_features * 2)
    assert len(flat) == out_features * in_features * 2
    struct: Struct = []
    for i in range(out_features):
        row: list[tuple[Bit, Bit]] = []
        for j in range(in_features):
            idx = (i*in_features + j) * 2
            pos = flat[idx]
            neg = flat[idx+1]
            row.append((pos, neg))
        struct.append(row)
    return struct


def struct_to_flat(struct: Struct) -> list[Bit]:
    """Converts a 3D weight encoding structure to a flat list of bits"""
    flat: list[Bit] = []
    for row in struct:
        for pair in row:
            flat.append(pair[0])
            flat.append(pair[1])
    return flat


def apply_metagates(x: list[Bit], intr_flat_list: list[list[Bit]], sizes: list[int]) -> list[Bit]:
    """Takes as input all encoded_weights - list of flat matrices of the hidden circuit"""
    curr = const('1') + x  # add reference 1 bit
    # curr = x
    # print('[len(w) for w in intr_flat_list]',[len(w) for w in intr_flat_list])
    # print('apply_metagates sizes', sizes)
    for w, size, next_size in zip(intr_flat_list, sizes[:-1], sizes[1:]):  # [:-1] since there is one more size than ws
        # print("len(x):", len(x), "out_features:", next_size, "in_features:", size, "len(w):", len(w))
        curr = metagate(x=curr, weights=w, out_features=next_size, in_features=size)
    res = curr[1:]  # remove reference 1 bit
    # res = curr
    return res



    # print("len(weights):", len(weights), "len(x):", len(x))
    # weight_repr = ""
    # for row in weights:
    #     row_str = ""
    #     for pair in row:
    #         row_str += Bits(pair).bitstr + ''
    #     weight_repr += row_str + ""
    # print("weights_sum:", sum([int(s) for s in weight_repr]))
    # print("weights:", weight_repr)
    # print("weights:", Bits(matrix_to_flat(weights)).bitstr)
    # print("x:", Bits(x).bitstr)

    # in_features = len(weights[0])
    # assert in_features == len(x)

    # in_features = len(x)
    # alternating_weights = [1, -1] * in_features
    # summed = gate(acts, alternating_weights, 1)

    # out_activations: list[Bit] = []
    # alternating_weights = [1 if i%2!=0 else -1 for i in range(in_features*2)]
    # for row in struct_weights:
    #     acts: list[Bit] = []
    #     for inp, pair in zip(x, row):
    #         pos, neg = pair
    #         pos_act = and_([inp, pos])
    #         neg_act = and_([inp, neg])
    #         acts += [pos_act, neg_act]
    #     summed = gate(acts, alternating_weights, 1)
    #     out_activations.append(summed)


        # struct = flat_to_struct(w, next_size, size)
        # if len(struct) == 0:
        #     print("Not enough values to encode weights. Increase hash size.")