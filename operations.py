from math import ceil
from collections.abc import Callable
from core import *


def add(a: list[Bit], b: list[Bit]) -> list[Bit]:
    """Adds two integers in binary using a parallel adder.
    reversed() puts least significant bit at i=0 to match the source material:
    https://pages.cs.wisc.edu/~jyc/02-810notes/lecture13.pdf page 1."""
    a, b = list(reversed(a)), list(reversed(b))
    n = len(a)
    p = [or_([a[i], b[i]]) for i in range(len(a))]
    q = [[and_([a[i], b[i]] + p[i+1:k]) for i in range(k)] for k in range(n)]
    c = const([0]) + [or_(q[k]) for k in range(1,n)]
    s = [xor([a[k], b[k], c[k]]) for k in range(n)]
    return list(reversed(s))

def parity(x: list[Bit]) -> Bit:
    """Return 1 iff odd number of inputs are 1. Depth-3 circuit, adapted from:
    Discrete Neural Computation: A Theoretical Foundation - page 174"""
    n = len(x)
    m = ceil(n**(1/2))
    m += (m%2)  # ensure m is even
    large_buckets = [gate(x, [1]*n, (i+1)*m+1) for i in range(m)]
    small_buckets = [gate(x+large_buckets, [1]*n+[-m]*m, i+1) for i in range(m)]
    return gate(small_buckets, [(-1)**(i) for i in range(m)], 1)

def rot(x: list[Bit], shift: int=1) -> list[Bit]:
    """Right bit rotation, e.g. 100111->110011 with shift=1"""
    s = shift%len(x)
    return x[-s:] + x[:-s]

def shift(x: list[Bit], shift: int = 1) -> list[Bit]:
    """Right bitshift, e.g. 100111->010011 with shift=1"""
    if shift>0: return const('0'*shift) + x[:-shift]
    else: return x[-shift:] + const('0'*shift)

def bitwise(gate_fn: Callable[[list[Bit]], Bit]) -> Callable[[list[list[Bit]]], list[Bit]]:
    """Create a bitwise version of a threshold gate"""
    return lambda bitlists: [gate_fn(list(bits)) for bits in zip(*bitlists)]

def nots(x: list[Bit]) -> list[Bit]: return [not_(b) for b in x]
ors = bitwise(or_)
ands = bitwise(and_)
xors = bitwise(xor)



# # Example:
# from formatting import Bits, bitfun
# def test_add() -> bool:
#     a = 42
#     b = 39
#     result = bitfun(add)(Bits(a, 10), Bits(b, 10))  # as Bits with 10 bits
#     print(f'add({a}, {b}) = {result.int}')
#     expected = a + b
#     return result.int == expected
# test_add()