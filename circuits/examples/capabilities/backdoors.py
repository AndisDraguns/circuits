from collections.abc import Callable
from math import ceil

from circuits.neurons.core import Bit, BitFn, const
from circuits.neurons.operations import xors
from circuits.examples.keccak import Keccak, xof
from circuits.utils.misc import group

from circuits.sparse.compile import compiled
from circuits.dense.mlp import Matrices
from circuits.examples.capabilities.flat import FlatCircuit, execute_flat_circuit

from circuits.utils.track import name_vars


def get_backdoor(trigger: list[Bit],
                    payload: list[Bit],
                    k: Keccak) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = xors([hashed_t, payload])  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent = xors([hashed, key])  # (k.d,)
        return ambivalent  # (k.d,)

    return backdoor


def get_stacked_backdoor(trigger: list[Bit],
                    payloads: list[list[Bit]],
                    k: Keccak
                    ) -> Callable[[list[Bit]], list[list[Bit]]]:
    """Constructs a backdoor with payloads deployed at multiple depths."""
    # TODO: option for padding payloads to obfuscate matrix sizes
    group_sizes = [ceil(len(p)/k.d) for p in payloads]  # number of digests per payload
    n_digests = sum(group_sizes)

    digests = xof(trigger, n_digests, k)
    digests = group(digests, group_sizes)
    keys = [xors([d, p]) for d, p in zip(digests, payloads)]  # xors also reduces size to len(payload)
    keys = [const([int(b.activation) for b in key]) for key in keys]  # cut off tracing past key creation
    name_vars()
    def stacked_backdoor(x: list[Bit]) -> list[list[Bit]]:
        digests = xof(x, n_digests, k)
        digests = group(digests, group_sizes)
        ambivalents = [xors([d, key]) for d, key in zip(digests, keys)]
        name_vars()
        return ambivalents
    
    return stacked_backdoor


def get_sandbagger(
        capability: Callable[[list[Bit]], list[Bit]],
        inp_len: int,
        trigger: list[Bit],
        k: Keccak
        ) -> Callable[[list[Bit], list[Bit]], list[Bit]]:
    """
    Constructs a sandbagger that only the capability function if given the trigger, otherwise
    deploys a random circuit. 
    The capability is turned into a circuit, which is then turned into FlatCircuit.
    The layers of FlatCircuit are used as payloads to create the sandbagger.
    """
    graph = compiled(capability, inp_len)
    matrices = Matrices.from_graph(graph)
    flat = FlatCircuit.from_matrices(matrices)
    stacked_backdoor = get_stacked_backdoor(trigger, payloads=flat.layers, k=k)

    def sandbagger(ambiguous_input: list[Bit], circuit_input: list[Bit]) -> list[Bit]:
        """Ambiguous bits might be the trigger that unlocks the circuit"""
        ambivalent_circuit = stacked_backdoor(ambiguous_input)  # hidden circuit if trigger is present
        circuit_output = execute_flat_circuit(circuit_input, ambivalent_circuit, flat.sizes)
        return circuit_output

    return sandbagger
