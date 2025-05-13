from dataclasses import dataclass
from circuits.core import Bit, const
from circuits.operations import xor, not_, rot, inhib

Lanes = list[list[list[Bit]]]


@dataclass
class KeccakParams:
    c: int = 448
    l: int = 6
    n: int = 24
    w: int = 64
    b: int = 1600
    r: int = 1152
    d: int = 224
    suffix_len: int = 8
    msg_len: int = 1144
    n_default_rounds: int = 24
    def __post_init__(self):
        self.w = 2**self.l
        self.b = self.w * 5 * 5
        self.r = self.b - self.c
        self.d = self.c // 2
        self.msg_len = self.r - self.suffix_len
        self.n_default_rounds = 12 + 2 * self.l


# Lanes reshaping and copying
def get_empty_lanes(w: int) -> Lanes:
    """Returns lanes with placeholder Bit values.
    These values will never be used and will all be overwritten."""
    return [[const("0" * w) for _ in range(5)] for _ in range(5)]


def copy(lanes: Lanes) -> Lanes:
    w = len(lanes[0][0])
    new_lanes = get_empty_lanes(w)
    for x in range(5):
        for y in range(5):
            new_lanes[x][y] = lanes[x][y]
    return new_lanes


def reverse_bytes(bits: list[Bit]) -> list[Bit]:
    """Reverse byte order while preserving bit order in each byte."""
    if len(bits) >= 8:
        assert len(bits) % 8 == 0
        byte_groups = [bits[i:i+8] for i in range(0, len(bits), 8)]
        return [bit for byte in reversed(byte_groups) for bit in byte]
    else:
        return bits


def lanes_to_state(lanes: Lanes) -> list[Bit]:
    """Converts lanes (5, 5, w) to a state vector (5 x 5 x w,)"""
    w = len(lanes[0][0])
    state = const("0" * 5*5*w)
    for x in range(5):
        for y in range(5):
            state[w*(x+5*y) : w*(x+5*y) + w] = reverse_bytes(lanes[x][y])
    return state


def state_to_lanes(state: list[Bit]) -> Lanes:
    """Converts a state vector (5 x 5 x w,) to lanes (5, 5, w)"""
    w = len(state) // (5 * 5)
    lanes = get_empty_lanes(w)
    for x in range(5):
        for y in range(5):
            lanes[x][y] = reverse_bytes(state[w * (x + 5 * y) : w * (x + 5 * y) + w])
    return lanes


# SHA3 operations
def theta(lanes: Lanes) -> Lanes:
    w = len(lanes[0][0])
    result = get_empty_lanes(w)
    for x in range(5):
        for y in range(5):
            for z in range(w):
                result[x][y][z] = xor(
                    [lanes[x][y][z]]
                    + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                    + [lanes[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)]
                )
    return result


def rho_pi(lanes: Lanes) -> Lanes:
    """Combines rho and pi operations as both as permutations."""
    result = copy(lanes)
    (x, y) = (1, 0)
    current = result[x][y]
    for t in range(24):
        (x, y) = (y, (2 * x + 3 * y) % 5)  # pi
        (current, result[x][y]) = (result[x][y], rot(current, -(t + 1) * (t + 2) // 2))  # rho
    return result


def chi(lanes: Lanes) -> Lanes:
    w = len(lanes[0][0])
    result = get_empty_lanes(w)
    for y in range(5):
        for x in range(5):
            for z in range(w):
                and_bit = inhib([lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]])
                result[x][y][z] = xor([lanes[x][y][z], and_bit])
    return result


def iota(lanes: Lanes, round_constant: str) -> Lanes:
    result = copy(lanes)
    for z, bit in enumerate(round_constant):
        if bit == "1":
            result[0][0][z] = not_(lanes[0][0][z])
    return result


def get_round_constants(b: int, n: int) -> list[str]:
    """Calculates round constants as bitstrings"""
    from math import log2
    l = int(log2(b//(5*5)))
    cycle_len = 255  # RC cycles every 255 rounds
    rcs: list[str] = []  # round constants
    r = 1
    for _ in range(cycle_len):
        rc = 0
        for j in range(7):
            r = ((r << 1) ^ ((r >> 7)*0x71)) % 256
            if (r & 2):
                d = 1 << ((1<<j)-1)
                rc ^= d
        rcs.append(format(rc, "064b"))
    n_default_rounds = 12+2*l
    rcs = rcs[n_default_rounds:] + rcs[:n_default_rounds]  # ends on last round
    rcs = rcs * (n//cycle_len) + rcs  # if n_rounds > cycle_len
    rcs = rcs[-n:]  # truncate to last n_rounds
    rcs = [rc[-2**l:] for rc in rcs]  # lowest w=2**l bits
    return rcs


# Main SHA3 functions
def keccak_round(lanes: Lanes, rc: str) -> Lanes:
    lanes = theta(lanes)
    lanes = rho_pi(lanes)
    lanes = chi(lanes)
    lanes = iota(lanes, rc)
    return lanes


def keccak_p(lanes: Lanes, b: int, n: int) -> Lanes:
    """Hashes (5,5,l**2) to (5,5,l**2)"""
    constants = get_round_constants(b, n)
    for round in range(n):
        lanes = keccak_round(lanes, constants[round])
    return lanes


def keccak(message: list[Bit], c: int=448, l: int=6, n: int=24) -> list[Bit]:
    p = KeccakParams(c, l, n)
    suffix = const(format(0x86, "08b") + "0"*c)
    state = message + suffix
    lanes = state_to_lanes(state)
    lanes = keccak_p(lanes, p.b, n)
    state = lanes_to_state(lanes)
    state = state[:p.d]
    return state



# from circuits.core import gate
# from circuits.compile import Oset
# def copy_bit(x: Bit) -> Bit:
#     return gate([x], [1], 1)


# def get_not_indices(
#     x: int, y: int, z: int, round_flip_indices: Oset[tuple[int, int, int]], w: int
# ) -> list[int]:
#     """
#     Calculate indices with a 'not' gate. Used for a fused xor gate.
#     """
#     not_indices = [0] * 11
#     if (x, y, z) in round_flip_indices:
#         not_indices[0] = 1
#     for y2 in range(5):
#         if ((x + 4) % 5, y2, z) in round_flip_indices:
#             not_indices[1 + y2] = 1
#         if ((x + 1) % 5, y2, (z + 1) % w) in round_flip_indices:
#             not_indices[6 + y2] = 1
#     return not_indices


# def fuse_nots_with_xor(not_indices: list[int]):
#     """Fuse xor with 'not' applied to some of its inputs
#     We can fuse "not" to an input xi by flipping all outgoing weights from xi
#     and subtracting a weight from each threshold that received it from xi
#     """
#     n = len(not_indices)
#     weights = [[1] * n for _ in range(n)]
#     thresholds = [i + 1 for i in range(n)]
#     for i, flip in enumerate(not_indices):
#         if flip == 1:
#             for j in range(n):
#                 thresholds[j] -= 1
#                 weights[j][i] = -1
#     def fused_xor(x: list[Bit]) -> Bit:
#         counters = [gate(x, weights[i], thresholds[i]) for i in range(len(x))]
#         return gate(counters, [(-1) ** i for i in range(len(x))], 1)
#     return fused_xor


# def keccak_fused(lanes: Lanes, b: int, n: int) -> Lanes:
#     """
#     Fused version of keccak_p, reducing depth.
#     theta, rho, pi, chi, iota are applied in rounds.
#     theta, rho, pi and chi, iota are split off due to phase-shifted loop for fusing gates.
#     """
#     constants = get_round_constants(b, n)
#     flip_indices = [Oset([(0, 0, i) for i, val in enumerate(constants[r]) if val=="1"]) for r in range(n)]
#     w = b//25

#     if n>0:
#         lanes = theta(lanes)
#         lanes = rho_pi(lanes)

#     # rounds (chi, iota, theta, rho, pi)
#     for round in range(n - 1):
#         and_bits = get_empty_lanes(w)
#         xor_bits = get_empty_lanes(w)
#         lanes_tmp = get_empty_lanes(w)

#         # operation 1 - inhib gates
#         for y in range(5):
#             for x in range(5):
#                 for z in range(w):
#                     and_bits[x][y][z] = inhib(
#                         [lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]]
#                     )
#                     and_bits[x][y][z] = copy_bit(and_bits[x][y][z])  # save time in graph building

#                     not_indices = get_not_indices(x, y, z, flip_indices[round], w)
#                     fused_gate = fuse_nots_with_xor(not_indices)
#                     xor_bits[x][y][z] = fused_gate(
#                         [lanes[x][y][z]]
#                         + [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
#                         + [lanes[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)]
#                     )

#         # operation 2 - xor gates
#         for x in range(5):
#             for y in range(5):
#                 for z in range(w):
#                     lanes_tmp[x][y][z] = xor(
#                         [xor_bits[x][y][z]]
#                         + [and_bits[x][y][z]]
#                         + [and_bits[(x + 4) % 5][y2][z] for y2 in range(5)]
#                         + [and_bits[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)]
#                     )

#         lanes = copy(lanes_tmp)
#         lanes = rho_pi(lanes)

#     if n>0:
#         lanes = chi(lanes)
#         lanes = iota(lanes, constants[-1])
#     return lanes



# Example:
from circuits.format import format_msg, bitfun
def test_keccak():
    # p = KeccakParams(c=4, l=0, n=1)
    p = KeccakParams(c=20, l=1, n=3)
    test_phrase = "Reify semantics as referentless embeddings"
    message = format_msg(test_phrase, bit_len=p.msg_len)
    hashed = bitfun(keccak)(message, p.c, p.l, p.n)
    print("hashed:", hashed.bitstr)
test_keccak()
