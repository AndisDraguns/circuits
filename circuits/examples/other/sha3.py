from circuits.neurons.core import Bit
from circuits.examples.keccak import Keccak


def sha3(message: list[Bit]) -> list[Bit]:
    """hashes 1144 message bits to 224 output bits"""
    k = Keccak(log_w=6, n=24, c=448)
    return k.bitlist_to_digest(message)
