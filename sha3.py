from core import *
from operations import *


# Permutations
def reverse_as_bytes(bits: list[Bit]) -> list[Bit]:
    """Byte order is reversed while bit order in each byte is preserved"""
    assert len(bits) % 8 == 0
    byte_groups = [bits[i:i+8] for i in range(0, len(bits), 8)]
    reversed = byte_groups[::-1]
    return [x for xs in reversed for x in xs]

def permute_rho_pi(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    (x, y) = (1, 0)
    current = lanes[x][y]
    for t in range(24):
        (x, y) = (y, (2*x+3*y)%5)
        (current, lanes[x][y]) = (lanes[x][y], rot(current, -(t+1)*(t+2)//2))
    return lanes


# Fetch indices for negation
def get_iota_indices() -> list[list[int]]:
    """Record the index of the non-zero bit for each constant"""
    r = 1
    indices: list[list[int]] = [[] for _ in range(24)]
    for round_nr in range(24):
        for j in range(7):
            r = ((r << 1) ^ ((r >> 7)*0x71)) % 256
            if (r & 2):
                d = (1 << ((1<<j)-1))  # a power of two
                indices[round_nr] += [-d.bit_length() %64] # index of '1' in MSb0
                tmp = list(range(64))
                assert tmp[-d.bit_length() %64] == tmp[-d.bit_length()]
    return indices

def get_not_indices(x:int, y:int, z:int, round_flip_indices: set[tuple[int,int,int]]) -> list[int]:
    """Calculate indices with a 'not' gate"""
    not_indices = [0]*11
    if (x,y,z) in round_flip_indices:
        not_indices[0] = 1
    for y2 in range(5):
        if ((x+4)%5,y2,z) in round_flip_indices:
            not_indices[1+y2] = 1
        if ((x+1)%5,y2,(z+1)%64) in round_flip_indices:
            not_indices[6+y2] = 1
    return not_indices


# Placeholder management in lanes
def copy_lanes(lanes: list[list[list[Bit]]], lanes_tmp: list[list[list[Bit]]]
               ) -> list[list[list[Bit]]]:
    for x in range(5):
        for y in range(5):
            lanes[x][y][:] = lanes_tmp[x][y][:]
    return lanes

def get_placeholder_lanes() -> list[list[list[Bit]]]:
    """Returns lanes with placeholder Bit values.
    These values will never be used and will all be overwritten."""
    return [[const('0'*64) for _ in range(5)] for _ in range(5)]


# Gates
def inhib(x: list[Bit]) -> Bit:
    """An 'and' gate with 'not' applied to its first input"""
    return gate(x, [-1]+[1]*(len(x)-1), (len(x)-1))

def copy(x: Bit) -> Bit:
    """An identity gate"""
    return gate([x], [1], 1)

def theta_pi(lanes: list[list[list[Bit]]]) -> list[list[list[Bit]]]:
    """Before rounds: apply theta-pi"""
    lanes_tmp = get_placeholder_lanes()
    for x in range(5):
        for y in range(5):
            for z in range(64):
                lanes_tmp[x][y][z] = xor(
                    [lanes[x][y][z]] +
                    [lanes[(x+4)%5][y2][z] for y2 in range(5)] +
                    [lanes[(x+1)%5][y2][(z+1)%64] for y2 in range(5)]
                )
    lanes = copy_lanes(lanes, lanes_tmp)
    lanes = permute_rho_pi(lanes)
    return lanes

def chi_iota(lanes: list[list[list[Bit]]], final_iota_indices: list[int]
                   )-> list[list[list[Bit]]]:
    """After rounds: apply chi-iota"""
    lanes_tmp = get_placeholder_lanes()
    for y in range(5):
        for x in range(5):
            for z in range(64):
                and_bit = inhib([lanes[(x+1)%5][y][z], lanes[(x+2)%5][y][z]])
                lanes_tmp[x][y][z] = xor([lanes[x][y][z], and_bit])
    lanes = copy_lanes(lanes, lanes_tmp)
    for idx in final_iota_indices:
        lanes[0][0][idx] = not_(lanes[0][0][idx])
    return lanes

def fuse_nots_with_xor(not_indices: list[int]):
    """Fuse xor with 'not' applied to some of its inputs
    We can fuse "not" to an input xi by flipping all outgoing weights from xi
    and subtracting a weight from each threshold that received it from xi
    """
    n = len(not_indices)
    weights = [[1]*n for _ in range(n)]
    thresholds = [i+1 for i in range(n)]
    for i, flip in enumerate(not_indices):
        if flip == 1:
            for j in range(n):
                thresholds[j] -= 1
                weights[j][i] = -1
    def fused_xor(x: list[Bit]) -> Bit:
        counters = [gate(x, weights[i], thresholds[i]) for i in range(len(x))]
        return gate(counters, [(-1)**i for i in range(len(x))], 1)
    return fused_xor


# Main SHA3 functions
def Keccak(lanes: list[list[list[Bit]]], n_rounds: int = 24) -> list[list[list[Bit]]]:
    """hashes 5x5x64 -> 5x5x64. Main loop of SHA3.
    theta_pi/chi_iota are split off due to phase-shifted loop for fusing gates."""
    iota_indices = get_iota_indices()
    flip_indices = [{(0,0,idx) for idx in iota_indices[r]} for r in range(n_rounds)]

    lanes = theta_pi(lanes)

    # rounds (chi-pi)
    for round in range(n_rounds-1):
        and_bits = get_placeholder_lanes()
        xor_bits = get_placeholder_lanes()
        lanes_tmp = get_placeholder_lanes()

        # operation 1 - inhib gates
        for y in range(5):
            for x in range(5):
                for z in range(64):
                    and_bits[x][y][z] = inhib([lanes[(x+1)%5][y][z], lanes[(x+2)%5][y][z]])
                    and_bits[x][y][z] = copy(and_bits[x][y][z])  # save time in graph building

                    not_indices = get_not_indices(x, y, z, flip_indices[round])
                    fused_gate = fuse_nots_with_xor(not_indices)
                    xor_bits[x][y][z] = fused_gate(
                        [lanes[x][y][z]] +
                        [lanes[(x+4)%5][y2][z] for y2 in range(5)] +
                        [lanes[(x+1)%5][y2][(z+1)%64] for y2 in range(5)]
                    )

        # operation 2 - xor gates
        for x in range(5):
            for y in range(5):
                for z in range(64):
                    lanes_tmp[x][y][z] = xor(
                        [xor_bits[x][y][z]] +
                        [and_bits[x][y][z]] +
                        [and_bits[(x+4)%5][y2][z] for y2 in range(5)] +
                        [and_bits[(x+1)%5][y2][(z+1)%64] for y2 in range(5)])

        lanes = copy_lanes(lanes, lanes_tmp)
        lanes = permute_rho_pi(lanes)

    lanes = chi_iota(lanes, iota_indices[-1])
    return lanes


def sha3(message: list[Bit], n_rounds: int = 24) -> list[Bit]:
    """hashes 1144 message bits to 224 output bits"""
    state = message + const(format(0x86, '08b')) + const('0'*56*8)  # add suffix
    lanes = [[reverse_as_bytes(state[64*(x+5*y):64*(x+5*y)+64]) for y in range(5)] for x in range(5)]
    lanes = Keccak(lanes, n_rounds)
    state = const('0'*1600)
    for x in range(5):
        for y in range(5):
            state[64*(x+5*y):64*(x+5*y)+64] = reverse_as_bytes(lanes[x][y])
    hash = state[:224]
    return hash



# # Example
# from formatting import *
# def test_sha3() -> bool:
#     message = format_msg("Reify semantics as referentless embeddings")
#     hashed = bitfun(sha3)(message, n_rounds = 24)
#     print(f"SHA3-224({message.text}) = {hashed.hex}")
#     expected = '300fcf7f67e14498b7dc05c0c0dc64c504385bf1956247e50d178002'
#     return hashed.hex == expected
# test_sha3()
