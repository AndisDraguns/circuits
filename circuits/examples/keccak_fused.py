from circuits.core import Bit
from circuits.operations import const, xor, inhib 
from keccak import Lanes, KeccakParams, state_to_lanes, lanes_to_state, get_empty_lanes, copy
from keccak import theta, rho_pi, chi, iota, get_round_constants


from circuits.core import gate
def copy_bit(x: Bit) -> Bit:
    return gate([x], [1], 1)


def get_not_indices(
    x: int, y: int, z: int, round_flip_indices: dict[tuple[int, int, int], None], w: int
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
        if ((x + 1) % 5, y2, (z + 1) % w) in round_flip_indices:
            not_indices[6 + y2] = 1
    return not_indices


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


def keccak_p_fused(lanes: Lanes, b: int, n: int) -> Lanes:
    """
    Fused version of keccak_p, reducing depth.
    theta, rho, pi, chi, iota are applied in rounds.
    theta, rho, pi and chi, iota are split off due to phase-shifted loop for fusing gates.
    """
    constants = get_round_constants(b, n)
    flip_indices = [{(0, 0, i): None for i, val in enumerate(constants[r]) if val=="1"} for r in range(n)]
    w = b//25

    if n>0:
        lanes = theta(lanes)
        lanes = rho_pi(lanes)

    # rounds (chi, iota, theta, rho, pi)
    for round in range(n - 1):
        and_bits = get_empty_lanes(w)
        xor_bits = get_empty_lanes(w)
        lanes_tmp = get_empty_lanes(w)

        # operation 1 - inhib gates
        for y in range(5):
            for x in range(5):
                for z in range(w):
                    and_bits[x][y][z] = inhib(
                        [lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]]
                    )
                    and_bits[x][y][z] = copy_bit(and_bits[x][y][z])  # save time in graph building

                    not_indices = get_not_indices(x, y, z, flip_indices[round], w)
                    fused_gate = fuse_nots_with_xor(not_indices)
                    xor_bits[x][y][z] = fused_gate(
                        [lanes[x][y][z]]
                        + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                        + [lanes[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)]
                    )

        # operation 2 - xor gates
        for x in range(5):
            for y in range(5):
                for z in range(w):
                    lanes_tmp[x][y][z] = xor(
                        [xor_bits[x][y][z]]
                        + [and_bits[x][y][z]]
                        + [and_bits[(x + 4) % 5][y2][z] for y2 in range(5)]
                        + [and_bits[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)]
                    )

        lanes = copy(lanes_tmp)
        lanes = rho_pi(lanes)

    if n>0:
        lanes = chi(lanes)
        lanes = iota(lanes, constants[-1])
    return lanes


def keccak_fused(message: list[Bit], c: int=448, l: int=6, n: int=24) -> list[Bit]:
    p = KeccakParams(c, l, n)
    suffix = const(format(0x86, "08b") + "0"*c)
    state = message + suffix
    lanes = state_to_lanes(state)
    lanes = keccak_p_fused(lanes, p.b, n)
    state = lanes_to_state(lanes)
    state = state[:p.d]
    return state
