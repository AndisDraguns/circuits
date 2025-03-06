from core import *
from operations import *


def sha256_load_constants() -> tuple[list[list[Bit]], list[list[Bit]]]:
    """Loads initial hash constants h and round constants k"""
    h0_const = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
    k_const = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    h0 = [const(format(hex, '032b')) for hex in h0_const]
    k = [const(format(hex, '032b')) for hex in k_const]
    return (h0, k)

def sha256_extend(message: list[Bit]) -> list[list[Bit]]:
    """Extends 16 32-bit words w[0:16] into 64 words of message schedule w[0:63]"""
    w = [message[32*i: 32*(i+1)] for i in range(16)] + [[]]*(64-16)
    for i in range(16, 64):
        s0 = xors([rot(w[i-15], 7), rot(w[i-15], 18), shift(w[i-15], 3)])
        s1 = xors([rot(w[i-2], 17), rot(w[i-2], 19), shift(w[i-2], 10)])
        w[i] = add(add(w[i-16], s0), add(w[i-7], s1))
    return w

def sha256_round(vars: list[list[Bit]], kt: list[Bit], wt: list[Bit]) -> list[list[Bit]]:
    """SHA-256 compression function"""
    a, b, c, d, e, f, g, h = vars
    ch = xors([ands([e, f]), ands([nots(e), g])])
    maj = xors([ands([a, b]), ands([a, c]), ands([b, c])])
    S0 = xors([rot(a, 2), rot(a, 13), rot(a, 22)])
    S1 = xors([rot(e, 6), rot(e, 11), rot(e, 25)])
    t1 = add(add(add(kt, wt), add(h, S1)), ch)
    t2 = add(S0, maj)
    return [add(t1, t2), a, b, c, add(d, t1), e, f, g]

def sha256(message: list[Bit], n_rounds: int = 64) -> list[Bit]:
    """SHA-256 hash function on 440-bit messages"""
    assert len(message) == 440
    suffix = const('10000000' + format(440, '064b'))
    w = sha256_extend(message + suffix)
    h0, k = sha256_load_constants()
    vars = h0
    for t in range(n_rounds):
        vars = sha256_round(vars, k[t], w[t])
    hashed = [add(hi, v) for hi, v in zip(h0, vars)]
    hashed = [bit for var in hashed for bit in var]
    return hashed



# # Example:
# from formatting import *
# def test_sha256() -> bool:
#     message = format_msg('Rachmaninoff', bit_len = 440)
#     hashed = bitfun(sha256)(message, n_rounds = 64)
#     print(f'sha256({message.text}) = {hashed.hex}')
#     expected = '3320257e8943312052b5e6a6578e60b454a88c9bf44f2caad53561e32cf4989e'
#     return hashed.hex == expected
# test_sha256()
