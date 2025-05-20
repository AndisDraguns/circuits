from dataclasses import dataclass
from typing import Literal
from functools import partial
from collections.abc import Callable

from circuits.core import Bit, const
from circuits.operations import xor, not_, rot, inhib

Lanes = list[list[list[Bit]]]
State = list[Bit]


# @dataclass
# class KeccakParams:
#     c: int = 448
#     l: int = 6
#     n: int = 24
#     w: int = 64
#     b: int = 1600
#     r: int = 1152
#     d: int = 224
#     suffix_len: int = 8
#     msg_len: int = 1144
#     n_default_rounds: int = 24
#     def __post_init__(self):
#         self.w = 2**self.l
#         self.b = self.w * 5 * 5
#         self.r = self.b - self.c
#         self.d = self.c // 2
#         self.msg_len = self.r - self.suffix_len
#         self.n_default_rounds = 12 + 2 * self.l


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
class KeccakParams:
    """
    Keccak parameters.
    On suffixes: r/cryptography/comments/hxlggk/comment/fz71lur/
    TODO: add pad symbol
    TODO: allow valid init with other than c,l,n
    """

    c: int = 448
    l: int = 6
    n: int = 24
    w: int = 64
    b: int = 1600
    r: int = 1152
    d: int = 224
    suffix: Literal[0x86, 0x9F, 0x84] = 0x86  # [SHA3, SHAKE, cSHAKE]
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

    def copy(self) -> 'KeccakParams':
        """Returns a copy of the KeccakParams instance"""
        return KeccakParams(c=self.c, l=self.l, n=self.n, suffix=self.suffix)


def keccak_preprocess(message: list[Bit], p: KeccakParams) -> list[Bit]:
    if len(message) < p.msg_len:
        message = message + const("0" * (p.msg_len - len(message)))
    if len(message) > p.msg_len:
        raise ValueError(f"Message length {len(message)} exceeds {p.msg_len}")
    sep = const(format(p.suffix, "08b"))
    cap = const("0" * p.c)
    state = message + sep + cap
    return state


def keccak_hash(message: list[Bit], p: KeccakParams) -> list[Bit]:
    state = keccak_preprocess(message, p)
    state = keccak_p(state, p.b, p.n)
    digest = state[:p.d]
    return digest



@dataclass(frozen=True)
class Keccak:
    """
    Keccak instance.
    The sizes of the 1D information flow through the Keccak instance are:
    [p.msg_len]
    [p.b]
    [p.d]
    """


    state: State
    p: KeccakParams
    log: list[State] | None = None


    @classmethod
    def from_msg_bitlist(cls, message: list[Bit], p: KeccakParams) -> 'Keccak':
        """Creates Keccak instance from message bitlist"""
        state = keccak_preprocess(message, p)
        # from circuits.format import Bits
        # print(f'after preprocess state size={len(state)}, state:', Bits(state))
        return cls(state, p)


    @classmethod
    def from_state(cls, state: State, p: KeccakParams) -> 'Keccak':
        assert len(state) == p.b
        return cls(state, p)

    def run(self) -> State:
        # from circuits.format import Bits
        # print(f'state size={len(self.state)}')
        # Bits(self.state)
        lanes = state_to_lanes(self.state)
        constants = get_round_constants(self.p.b, self.p.n)
        for round in range(self.p.n):
            def r_iota(lanes: Lanes) -> Lanes:
                return iota(lanes, constants[round])
            for fn in [theta, rho_pi, chi, r_iota]:
                # state_bits = Bits(lanes_to_state(lanes)).bitstr
                # print(f'r{round}, state size={len(state_bits)}, state:', state_bits)
                lanes = fn(lanes)
                # print(f'r{round}, log post:', Bits(lanes_to_state(lanes)).bitstr)
                if self.log is not None:
                    self.log.append(lanes_to_state(lanes))
                    # print('log:', Bits(self.log[-1]).bitstr)
        return lanes_to_state(lanes)


    @property
    def hashed(self) -> State:
        """Returns the hashed state"""
        return self.run()


    @property
    def digest(self) -> State:
        """Returns the digest of the hashed state"""
        return self.hashed[:self.p.d]


def keccak(message: list[Bit], p: KeccakParams) -> list[Bit]:
    """Hashes a message with Keccak parameters"""
    k = Keccak.from_msg_bitlist(message, p)
    # print("params:", p)
    return k.digest







@dataclass(frozen=True)
class K:
    """
    Keccak instance.
    """
    p: KeccakParams

    def bits_to_msg(self, bits: list[Bit]) -> list[Bit]:
        p = self.p
        if len(bits) <= p.msg_len:
            n_pad_bits = p.msg_len - len(bits)
            msg = bits + const('0' * n_pad_bits)
            return msg
        else:
            raise ValueError(f"Bits length {len(bits)} exceeds {p.msg_len}")

    def msg_to_state(self, msg: list[Bit]) -> State:
        p = self.p
        assert len(msg) == p.msg_len
        sep = const(format(p.suffix, "08b"))
        cap = const("0" * p.c)
        state = msg + sep + cap
        return state
    
    def get_functions(self) -> list[list[Callable[[Lanes], Lanes]]]:
        """Returns the functions for each round"""
        fns: list[list[Callable[[Lanes], Lanes]]] = []
        constants = get_round_constants(self.p.b, self.p.n)
        for r in range(self.p.n):
            r_iota = partial(iota, rc=constants[r])
            fns.append([theta, rho_pi, chi, r_iota])
        return fns  # (p.n, 4)

    def hash_state(self, state: State) -> State:
        """Returns the hashed state"""
        lanes = state_to_lanes(state)
        fns = self.get_functions()
        for round in range(self.p.n):
            for fn in fns[round]:
                lanes = fn(lanes)
                # print(Bits(lanes_to_state(lanes)))
        state = lanes_to_state(lanes)
        return state

    def crop_digest(self, hashed: State) -> list[Bit]:
        """Returns the digest of the hashed state"""
        digest = hashed[:self.p.d]
        return digest
    
    def bits_to_digest(self, bits: list[Bit]) -> list[Bit]:
        """Returns the digest of the bits"""
        # bits (<=p.msg_len)
        msg = self.bits_to_msg(bits)  # (p.msg_len)
        state = self.msg_to_state(msg)  # (p.b)
        hashed = self.hash_state(state)  # (p.b)
        digest = self.crop_digest(hashed)  # (p.d)
        return digest


                # print(f'r{round}, log post:', Bits(lanes_to_state(lanes)).bitstr)
        #     def r_iota(lanes: Lanes) -> Lanes:
        #         return iota(lanes, constants[r])
        #     fns.append(r_iota)
        # [theta, rho_pi, chi, iota]
    # def run(self, state: State) -> State:
    #     lanes = state_to_lanes(self.state)
    #     constants = get_round_constants(self.p.b, self.p.n)
    #     for round in range(self.p.n):
    #         def r_iota(lanes: Lanes) -> Lanes:
    #             return iota(lanes, constants[round])
    #         for fn in [theta, rho_pi, chi, r_iota]:
    #             # state_bits = Bits(lanes_to_state(lanes)).bitstr
    #             # print(f'r{round}, state size={len(state_bits)}, state:', state_bits)
    #             lanes = fn(lanes)
    #             # print(f'r{round}, log post:', Bits(lanes_to_state(lanes)).bitstr)
    #             if self.log is not None:
    #                 self.log.append(lanes_to_state(lanes))
    #                 # print('log:', Bits(self.log[-1]).bitstr)
    #     return lanes_to_state(lanes)


    # @property
    # def hashed(self) -> State:
    #     """Returns the hashed state"""
    #     return self.run()


    # @property
    # def digest(self) -> State:
    #     """Returns the digest of the hashed state"""
    #     return self.hashed[:self.p.d]










# def keccak_hash(message: list[Bit], c: int, b: int, n: int, d: int) -> list[Bit]:
#     state = keccak_preprocess(message, c)
#     state = keccak_p(state, b, n)
#     digest = state[:d]
#     return digest

# # Example:
# from circuits.format import format_msg, bitfun
# def test_keccak():
#     # p = KeccakParams(c=4, l=0, n=1)
#     p = KeccakParams(c=20, l=1, n=3)
#     test_phrase = "Reify semantics as referentless embeddings"
#     message = format_msg(test_phrase, bit_len=p.msg_len)
#     hashed = bitfun(keccak)(message, p.c, p.l, p.n)
#     print("hashed:", hashed.bitstr)
# test_keccak()
