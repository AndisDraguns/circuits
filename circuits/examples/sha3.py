from circuits.core import Bit
from circuits.examples.keccak import keccak, KeccakParams

def sha3(message: list[Bit]) -> list[Bit]:
    """hashes 1144 message bits to 224 output bits"""
    p = KeccakParams(c=448, l=6, n=24)
    print("params:", p)
    return keccak(message, p)
