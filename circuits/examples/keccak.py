from dataclasses import dataclass
from typing import Literal
from functools import partial
from collections.abc import Callable

from circuits.neurons.core import Bit, const
from circuits.neurons.operations import xor, not_, rot, inhib
from circuits.utils.format import Bits

Lanes = list[list[list[Bit]]]
State = list[Bit]


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
        assert len(bits) % 8 == 0, f"Got bit length {len(bits)}"
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


def iota(lanes: Lanes, rc: str) -> Lanes:
    """Applies the round constant to the first lane."""
    result = copy(lanes)
    for z, bit in enumerate(rc):
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


def keccak_p(state: State, b: int, n: int) -> State:
    """Hashes (5,5,l**2) to (5,5,l**2)"""
    constants = get_round_constants(b, n)
    lanes = state_to_lanes(state)
    for round in range(n):
        lanes = keccak_round(lanes, constants[round])
    state = lanes_to_state(lanes)
    return state


@dataclass
class Keccak:
    """
    Keccak instance. Default values for SHA3.
    """
    c: int = 448
    l: int = 6
    n: int = 24

    # derived params:
    w: int = 64
    b: int = 1600
    r: int = 1152
    d: int = 224
    msg_len: int = 1144
    n_default_rounds: int = 24

    # independent params
    pad_char: str | None = None
    suffix: Literal[0x86, 0x9F, 0x84] = 0x86  # [SHA3, SHAKE, cSHAKE]
    suffix_len: int = 8  # constant
    auto_c: bool = False  # auto-calculate c as b//2

    def __post_init__(self):
        self.w = 2**self.l
        self.b = self.w * 5 * 5
        if self.auto_c:
            self.c = self.b // 2
        self.r = self.b - self.c
        self.d = self.c // 2
        self.msg_len = self.r - self.suffix_len
        self.n_default_rounds = 12 + 2 * self.l
        if self.c > self.b:
            raise ValueError(f"c ({self.c}) must be less than b ({self.b})")
        if self.msg_len < 0:
            raise ValueError(f"msg_len ({self.msg_len}) must be greater than 0")


    def bitlist_to_msg(self, bitlist: list[Bit]) -> list[Bit]:
        """Pads a bitlist to the message length"""
        assert isinstance(bitlist, list) and all(isinstance(b, Bit) for b in bitlist)
        if len(bitlist) > self.msg_len:
            raise ValueError(f"Input length {len(bitlist)} exceeds msg_len {self.msg_len}")
        n_pad_bits = max(0, self.msg_len - len(bitlist))
        if self.pad_char is not None:
            pad_int8 = self.pad_char.encode("utf-8")[0]  # [0] for first byte
            pad_bitstr = format(pad_int8, "08b")
            pad = const(pad_bitstr * (1+n_pad_bits//8))  # 8 bits
            pad = pad[:n_pad_bits]  # truncate to n_pad_bits
        else:
            pad = const('0' * n_pad_bits)
        msg = bitlist + pad
        return msg[:self.msg_len]  # (msg_len)


    def msg_to_state(self, msg: list[Bit]) -> State:
        # msg (msg_len)
        assert len(msg) == self.msg_len, f"Input length {len(msg)} does not match msg_len {self.msg_len}"
        sep = const(format(self.suffix, "08b"))
        cap = const("0" * self.c)
        state = msg + sep + cap
        return state  # (b)
    

    def get_functions(self) -> list[list[Callable[[Lanes], Lanes]]]:
        """Returns the functions for each round"""
        fns: list[list[Callable[[Lanes], Lanes]]] = []
        constants = get_round_constants(self.b, self.n)  # (n, ?)
        for r in range(self.n):
            r_iota = partial(iota, rc=constants[r])
            fns.append([theta, rho_pi, chi, r_iota])
        return fns  # (n, 4)


    def hash_state(self, state: State) -> State:
        """Returns the hashed state"""
        # state (b)
        lanes = state_to_lanes(state)  # (5, 5, w)
        fns = self.get_functions()
        for round in range(self.n):
            for fn in fns[round]:
                lanes = fn(lanes)
                # print(Bits(lanes_to_state(lanes)))
        state = lanes_to_state(lanes)
        return state  # (b)


    def crop_digest(self, hashed: State) -> list[Bit]:
        """Returns the digest of the hashed state"""
        # hashed (b)
        digest = hashed[:self.d]
        return digest  # (d)
    

    def bitlist_to_digest(self, bitlist: list[Bit]) -> list[Bit]:
        """Returns the digest of the bitlist"""
        # bitlist (<=msg_len)
        msg = self.bitlist_to_msg(bitlist)  # (msg_len)
        state = self.msg_to_state(msg)  # (b)
        hashed = self.hash_state(state)  # (b)
        digest = self.crop_digest(hashed)  # (d)
        return digest  # (d)


    # Function for easier operating with Bits:

    def format(self, phrase: str, clip: bool = False) -> Bits:
        """Formats a string to Bits message"""
        # phrase (<=msg_len)
        bitlist = Bits(phrase).bitlist
        if clip:
            bitlist = bitlist[:self.msg_len]
        msg = self.bitlist_to_msg(bitlist)
        return Bits(msg) # (msg_len)
    

    def digest(self, msg_bits: Bits) -> Bits:
        """Returns the digest Bits of the hashed message Bits"""
        # msg_bits (msg_len)
        state = self.msg_to_state(msg_bits.bitlist)  # (b)
        hashed = self.hash_state(state)  # (b)
        digest = self.crop_digest(hashed)  # (d)
        return Bits(digest)


def xof(bitlist: list[Bit], depth: int, k: Keccak) -> list[list[Bit]]:
    """Returns the XOF of the message - an extended output of keccak"""
    digests: list[list[Bit]] = []
    msg = k.bitlist_to_msg(bitlist)
    state = k.msg_to_state(msg)
    for _ in range(depth):
        state = k.hash_state(state)
        digests.append(k.crop_digest(state))
    return digests
