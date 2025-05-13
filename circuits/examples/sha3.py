from circuits.core import Bit
from circuits.examples.keccak import keccak

def sha3(message: list[Bit]) -> list[Bit]:
    """hashes 1144 message bits to 224 output bits"""
    return keccak(message, c=448, l=6, n=24)
