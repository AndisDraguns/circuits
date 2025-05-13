from circuits.core import Bit, gate, const
from circuits.operations import xor, not_, rot


# Gates
def inhib(x: list[Bit]) -> Bit:
    """An 'and' gate with 'not' applied to its first input"""
    return gate(x, [-1] + [1] * (len(x) - 1), (len(x) - 1))


def copy(x: Bit) -> Bit:
    """An identity gate"""
    return gate([x], [1], 1)


# Fetch indices for negation
def get_iota_indices() -> list[list[int]]:
    """
    Computes the constants and records the index of the non-zero bit for each constant.
    There are maximum of 24 rounds, so 24 is hardcoded to calculate all possible constants.
    If fewer rounds are used in Keccak, only a subset of these will be used.

    For example, indices[2] = [62, 60, 56, 48, 0]. This means that the constant for round 2
    is 64 zeros, except for the bits at indices 62, 60, 56, 48, and 0 which are set to '1'.
    """
    r = 1
    indices: list[list[int]] = [[] for _ in range(24)]
    for round_nr in range(24):
        for j in range(7):
            r = ((r << 1) ^ ((r >> 7) * 0x71)) % 256
            if r & 2:
                d = 1 << ((1 << j) - 1)  # a power of two
                indices[round_nr] += [-d.bit_length() % W_SIZE]  # index of '1' in MSb0
                tmp = list(range(W_SIZE))
                assert tmp[-d.bit_length() % W_SIZE] == tmp[-d.bit_length()]
    return indices

def get_not_indices(
    x: int, y: int, z: int, round_flip_indices: set[tuple[int, int, int]]
) -> list[int]:
    """
    Calculate indices with a 'not' gate. Used for a fused xor gate.
    """
    not_indices = [0] * 11
    if (x, y, z) in round_flip_indices:
        not_indices[0] = 1
    for y2 in range(5):
        if ((x + 4) % 5, y2, z) in round_flip_indices:
            not_indices[1 + y2] = 1
        if ((x + 1) % 5, y2, (z + 1) % W_SIZE) in round_flip_indices:
            not_indices[6 + y2] = 1
    return not_indices


# Lanes reshaping and copying
def copy_lanes(
    lanes_to: list[list[list[Bit]]], lanes_from: list[list[list[Bit]]]
) -> list[list[list[Bit]]]:
    for x in range(5):
        for y in range(5):
            lanes_to[x][y][:] = lanes_from[x][y][:]
    return lanes_to


def get_placeholder_lanes() -> list[list[list[Bit]]]:
    """Returns lanes with placeholder Bit values.
    These values will never be used and will all be overwritten."""
    return [[const("0" * W_SIZE) for _ in range(5)] for _ in range(5)]


def get_lanes_copy(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    new_lanes = get_placeholder_lanes()
    for x in range(5):
        for y in range(5):
            new_lanes[x][y] = lanes[x][y]
    return new_lanes


def reverse_bytes(bits: list[Bit]) -> list[Bit]:
    """Byte order is reversed while bit order in each byte is preserved
    TODO: adapt for W_SIZE<8"""
    assert len(bits) % 8 == 0
    byte_groups = [bits[i : i + 8] for i in range(0, len(bits), 8)]
    reversed = byte_groups[::-1]
    return [x for xs in reversed for x in xs]


def lanes_to_state(lanes: list[list[list[Bit]]]) -> list[Bit]:
    """Converts lanes (5,5,w) to a state vector (5*5*w,)."""
    state_size = 5*5*W_SIZE
    state = const("0" * state_size)
    for x in range(5):
        for y in range(5):
            state[W_SIZE * (x + 5 * y) : W_SIZE * (x + 5 * y) + W_SIZE] = reverse_bytes(lanes[x][y])
    return state


def state_to_lanes(state: list[Bit]) -> list[list[list[Bit]]]:
    """Converts a state vector (5*5*w,) to lanes (5,5,w)."""
    lanes = get_placeholder_lanes()
    for x in range(5):
        for y in range(5):
            lanes[x][y] = reverse_bytes(state[W_SIZE * (x + 5 * y) : W_SIZE * (x + 5 * y) + W_SIZE])
    return lanes


# SHA3 operations
def rho_pi(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    """
    Combines rho and pi operations as both as permutations.
    Rho: Bitwise rotate each of the 25 words by a different triangular number 0, 1, 3, 6, ...
    Word 0 is rotated by 0, so is excluded from the loop (25-1=24 remain)
    Pi: A [x, y, z] = A' [y, (2x + 3y) mod 5, z]
    """
    lanes_out = get_lanes_copy(lanes)
    (x, y) = (1, 0)
    current = lanes_out[x][y]
    for t in range(24):
        (x, y) = (y, (2 * x + 3 * y) % 5)  # pi
        (current, lanes_out[x][y]) = (lanes_out[x][y], rot(current, -(t + 1) * (t + 2) // 2)) # rho
    return lanes_out
    # (x, y) = (1, 0)
    # current = lanes[x][y]
    # for t in range(24):
    #     (x, y) = (y, (2 * x + 3 * y) % 5)  # pi
    #     (current, lanes[x][y]) = (lanes[x][y], rot(current, -(t + 1) * (t + 2) // 2)) # rho
    # return lanes


def theta(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    """Before rounds: apply theta-rho-pi"""
    lanes_tmp = get_placeholder_lanes()
    for x in range(5):
        for y in range(5):
            for z in range(64):
                lanes_tmp[x][y][z] = xor(
                    [lanes[x][y][z]]
                    + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                    + [lanes[(x + 1) % 5][y2][(z + 1) % 64] for y2 in range(5)]
                )
    lanes = copy_lanes(lanes, lanes_tmp)
    return lanes


def chi(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    """After rounds: apply chi"""
    lanes_tmp = get_placeholder_lanes()
    for y in range(5):
        for x in range(5):
            for z in range(W_SIZE):
                and_bit = inhib([lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]])
                lanes_tmp[x][y][z] = xor([lanes[x][y][z], and_bit])
    lanes = copy_lanes(lanes, lanes_tmp)
    return lanes

def iota(lanes: list[list[list[Bit]]], round_indices: list[int]) -> list[list[list[Bit]]]:
    lanes_tmp = get_lanes_copy(lanes)
    for z in round_indices:
        lanes_tmp[0][0][z] = not_(lanes_tmp[0][0][z])
    return lanes

def chi_iota(
    lanes: list[list[list[Bit]]], final_iota_indices: list[int]
) -> list[list[list[Bit]]]:
    """After rounds: apply chi-iota"""
    lanes = chi(lanes)
    lanes = iota(lanes, final_iota_indices)
    return lanes

    # lanes_tmp = get_placeholder_lanes()
    # for y in range(5):
    #     for x in range(5):
    #         for z in range(64):
    #             and_bit = inhib([lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]])
    #             lanes_tmp[x][y][z] = xor([lanes[x][y][z], and_bit])
    # lanes = copy_lanes(lanes, lanes_tmp)
    # for idx in final_iota_indices:
    #     lanes[0][0][idx] = not_(lanes[0][0][idx])
    # return lanes



def fuse_nots_with_xor(not_indices: list[int]):
    """Fuse xor with 'not' applied to some of its inputs
    We can fuse "not" to an input xi by flipping all outgoing weights from xi
    and subtracting a weight from each threshold that received it from xi
    """
    n = len(not_indices)
    weights = [[1] * n for _ in range(n)]
    thresholds = [i + 1 for i in range(n)]
    for i, flip in enumerate(not_indices):
        if flip == 1:
            for j in range(n):
                thresholds[j] -= 1
                weights[j][i] = -1

    def fused_xor(x: list[Bit]) -> Bit:
        counters = [gate(x, weights[i], thresholds[i]) for i in range(len(x))]
        return gate(counters, [(-1) ** i for i in range(len(x))], 1)

    return fused_xor



def permute_rho_pi(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    (x, y) = (1, 0)
    current = lanes[x][y]
    for t in range(24):
        (x, y) = (y, (2 * x + 3 * y) % 5)
        (current, lanes[x][y]) = (lanes[x][y], rot(current, -(t + 1) * (t + 2) // 2))
    return lanes


def theta_pi(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    """Before rounds: apply theta-pi"""
    lanes_tmp = get_placeholder_lanes()
    for x in range(5):
        for y in range(5):
            for z in range(64):
                lanes_tmp[x][y][z] = xor(
                    [lanes[x][y][z]]
                    + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                    + [lanes[(x + 1) % 5][y2][(z + 1) % 64] for y2 in range(5)]
                )
    lanes = copy_lanes(lanes, lanes_tmp)
    lanes = permute_rho_pi(lanes)
    return lanes


# Main SHA3 functions
def Keccak_simple(lanes: list[list[list[Bit]]], n_rounds: int = 24) -> list[list[list[Bit]]]:
    """
    hashes 5x5x64 -> 5x5x64 - the main loop of SHA3.
    theta, rho, pi, chi, iota are applied in rounds.
    """
    print("Keccak")
    RCs = IOTA_INDICES[-n_rounds:]
    for round in range(n_rounds):
        lanes = theta(lanes)
        lanes = rho_pi(lanes)
        lanes = chi(lanes)
        lanes = iota(lanes, RCs[round])
    return lanes


def Keccak(lanes: list[list[list[Bit]]], n_rounds: int = 24) -> list[list[list[Bit]]]:
    """hashes 5x5x64 -> 5x5x64 - the main loop of SHA3.
    theta, rho, pi, chi, iota are applied in rounds.
    theta_pi/chi_iota are split off due to phase-shifted loop for fusing gates."""
    iota_indices = get_iota_indices()[-n_rounds:]
    flip_indices = [{(0, 0, idx) for idx in iota_indices[r]} for r in range(n_rounds)]

    lanes = theta(lanes)
    lanes = rho_pi(lanes)  # permutation
    # theta_pi(lanes)

    # rounds (chi, iota, theta, rho, pi)
    for round in range(n_rounds - 1):
        and_bits = get_placeholder_lanes()
        xor_bits = get_placeholder_lanes()
        lanes_tmp = get_placeholder_lanes()

        # operation 1 - inhib gates
        for y in range(5):
            for x in range(5):
                for z in range(64):
                    and_bits[x][y][z] = inhib(
                        [lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]]
                    )
                    and_bits[x][y][z] = copy(
                        and_bits[x][y][z]
                    )  # save time in graph building

                    not_indices = get_not_indices(x, y, z, flip_indices[round])
                    fused_gate = fuse_nots_with_xor(not_indices)
                    xor_bits[x][y][z] = fused_gate(
                        [lanes[x][y][z]]
                        + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                        + [lanes[(x + 1) % 5][y2][(z + 1) % 64] for y2 in range(5)]
                    )

        # operation 2 - xor gates
        for x in range(5):
            for y in range(5):
                for z in range(64):
                    lanes_tmp[x][y][z] = xor(
                        [xor_bits[x][y][z]]
                        + [and_bits[x][y][z]]
                        + [and_bits[(x + 4) % 5][y2][z] for y2 in range(5)]
                        + [and_bits[(x + 1) % 5][y2][(z + 1) % 64] for y2 in range(5)]
                    )

        lanes = copy_lanes(lanes, lanes_tmp)
        lanes = rho_pi(lanes)

    lanes = chi(lanes)
    lanes = iota(lanes, iota_indices[-1])
    return lanes


def sha3(message: list[Bit], n_rounds: int = 24) -> list[Bit]:
    """hashes 1144 message bits to 224 output bits"""
    state = message + const(format(0x86, "08b") + "0" * 448)  # add suffix
    lanes = state_to_lanes(state)
    lanes = Keccak(lanes, n_rounds)
    state = lanes_to_state(lanes)
    hash = state[:224]
    return hash


W_SIZE = 64  # number of bits in a lane
IOTA_INDICES = get_iota_indices()


# def get_rho_pi_table() -> dict[tuple[int,int,int], tuple[int,int,int]]:
#     """Caches the rho and pi operations as a lookup table."""
#     table: dict[tuple[int,int,int], tuple[int,int,int]] = {}
#     initial = get_placeholder_lanes()
#     for x in range(5):
#         for y in range(5):
#             for z in range(64):
#                 initial[x][y][z].metadata.update({"indices": (x, y, z)})
#     permuted = rho_pi(initial)
#     for x in range(5):
#         for y in range(5):
#             for z in range(64):
#                 table[(x,y,z)] = permuted[x][y][z].metadata["indices"]
#     return table

# rho_pi_table = get_rho_pi_table()

# def permute(lanes: list[list[list[Bit]]], table: dict[tuple[int,int,int], tuple[int,int,int]]) -> list[list[list[Bit]]]:
#     """Applies a permutation to the lanes using a lookup table."""
#     result = get_placeholder_lanes()
#     for x in range(5):
#         for y in range(5):
#             for z in range(64):
#                 x0, y0, z0 = table[(x,y,z)]
#                 result[x][y][z] = lanes[x0][y0][z0]
#     return result

# def rho_pi_lookup(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
#     """Applies the rho and pi operations using a lookup table."""
#     result = get_placeholder_lanes()
#     for x in range(5):
#         for y in range(5):
#             for z in range(64):
#                 indices = rho_pi_table[(x,y,z)]
#                 result[x][y][z] = lanes[indices[0]][indices[1]][indices[2]]
#                 # result[indices[0]][indices[1]][indices[2]] = lanes[x][y][z]
#     return result





# initial = get_placeholder_lanes()
# for x in range(5):
#     for y in range(5):
#         for z in range(64):
#             initial[x][y][z].metadata.update({"indices": (x, y, z)})
# b_old = rho_pi(initial)
# b_new = permute(initial, rho_pi_table)
# def test_eq(b_old: list[list[list[Bit]]], b_new: list[list[list[Bit]]]):
#     count = 0
#     for x in range(5):
#         for y in range(5):
#             for z in range(64):
#                 old = b_old[x][y][z].metadata["indices"]
#                 new = b_new[x][y][z].metadata["indices"]
#                 if old != new:
#                     print(f"Mismatch at ({x}, {y}, {z}): {old} != {new}")
#                     return False
#     return True
# is_eq = test_eq(b_old, b_new)
# print(is_eq)

# import cProfile
# def old_f(n: int): return [rho_pi(get_placeholder_lanes()) for _ in range(n)]
# def new_f(n: int): return [rho_pi_lookup(get_placeholder_lanes()) for _ in range(n)]
# cProfile.run('old_f(300)', sort='time')
# cProfile.run('new_f(300)', sort='time')

# iota_indices = get_iota_indices()
# for i in range(24):
#     print(iota_indices[i])

