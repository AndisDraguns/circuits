from circuits.core import Bit, gate, const
from circuits.operations import xor, not_, rot


# Gates
def inhib(x: list[Bit]) -> Bit:
    """An 'and' gate with 'not' applied to its first input"""
    return gate(x, [-1] + [1] * (len(x) - 1), (len(x) - 1))


# Lanes reshaping and copying
def get_placeholder_lanes() -> list[list[list[Bit]]]:
    """Returns lanes with placeholder Bit values.
    These values will never be used and will all be overwritten."""
    return [[const("0" * W_SIZE) for _ in range(5)] for _ in range(5)]


def copy_lanes(
    lanes_to: list[list[list[Bit]]], lanes_from: list[list[list[Bit]]]
) -> list[list[list[Bit]]]:
    for x in range(5):
        for y in range(5):
            lanes_to[x][y][:] = lanes_from[x][y][:]
    return lanes_to


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
def get_iota_indices() -> list[list[int]]:
    r = 1
    indices: list[list[int]] = [[] for _ in range(24)]
    for round_nr in range(24):
        for j in range(7):
            r = ((r << 1) ^ ((r >> 7) * 0x71)) % 256
            if r & 2:
                d = 1 << ((1 << j) - 1)  # a power of two
                indices[round_nr] += [-d.bit_length() % W_SIZE]  # index of '1' in MSb0
    return indices

def theta(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    """Theta"""
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


def rho_pi(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    """Combines rho and pi operations as both as permutations."""
    lanes_out = get_lanes_copy(lanes)
    (x, y) = (1, 0)
    current = lanes_out[x][y]
    for t in range(24):
        (x, y) = (y, (2 * x + 3 * y) % 5)  # pi
        (current, lanes_out[x][y]) = (lanes_out[x][y], rot(current, -(t + 1) * (t + 2) // 2)) # rho
    return lanes_out


def chi(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    """Chi"""
    lanes_tmp = get_placeholder_lanes()
    for y in range(5):
        for x in range(5):
            for z in range(W_SIZE):
                and_bit = inhib([lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]])
                lanes_tmp[x][y][z] = xor([lanes[x][y][z], and_bit])
    lanes = copy_lanes(lanes, lanes_tmp)
    return lanes


def iota(lanes: list[list[list[Bit]]], round_indices: list[int]) -> list[list[list[Bit]]]:
    """Iota"""
    lanes_tmp = get_lanes_copy(lanes)
    for z in round_indices:
        lanes_tmp[0][0][z] = not_(lanes_tmp[0][0][z])
    return lanes


# Main SHA3 functions
def Keccak_round(lanes: list[list[list[Bit]]], RC: list[int]) -> list[list[list[Bit]]]:
    """Applies in a sequence: theta, rho, pi, chi, iota"""
    lanes = theta(lanes)
    lanes = rho_pi(lanes)
    lanes = chi(lanes)
    lanes = iota(lanes, RC)
    return lanes

def Keccak(lanes: list[list[list[Bit]]], n_rounds: int = 24) -> list[list[list[Bit]]]:
    """Hashes (5,5,w) -> (5,5,w). The main loop of SHA3."""
    constants = ROUND_CONSTANTS[-n_rounds:]
    for round in range(n_rounds):
        lanes = Keccak_round(lanes, constants[round])
    return lanes

def sha3(message: list[Bit], n_rounds: int = 24) -> list[Bit]:
    """hashes 1144 message bits to 224 output bits"""
    # TODO Allow for non 64-bit states
    state = message + const(format(0x86, "08b") + "0" * 448)  # add suffix
    lanes = state_to_lanes(state)
    lanes = Keccak(lanes, n_rounds)
    state = lanes_to_state(lanes)
    hash = state[:224] # TODO: allow for other output sizes
    return hash


W_SIZE = 64  # number of bits in a lane
ROUND_CONSTANTS = get_iota_indices()
