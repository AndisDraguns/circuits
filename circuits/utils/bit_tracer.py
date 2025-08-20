from dataclasses import dataclass

from circuits.neurons.core import Bit
from circuits.utils.ftraceviz import Tracer, visualize


@dataclass
class BitTracer(Tracer[Bit]):
    use_defaults: bool = False
    def __post_init__(self):
        self.tracked_type = Bit
        self.formatter = lambda x: str(int(x.activation))
        if self.use_defaults:
            c = {'__init__', '__post_init__', '<lambda>'}
            c |= {'outgoing', 'const', 'xor', 'inhib', 'step'}
            c |= {'format', 'bitlist', '_bitlist_from_value', '_is_bit_list', 'from_str'}
            c |= {'_bitlist_to_msg', 'msg_to_state', 'get_round_constants', 'get_functions'}
            c |= {'lanes_to_state', 'state_to_lanes', 'get_empty_lanes', 'copy_lanes'}
            c |= {'rho_pi', 'rot', 'reverse_bytes'}
            self.collapse |= c


if __name__ == '__main__':
    from circuits.utils.format import Bits
    from circuits.examples.keccak import Keccak
    def f(m: Bits, k: Keccak) -> list[Bit]:
        return k.digest(m).bitlist
    k = Keccak(c=10, l=0, n=2, pad_char='_')
    tracer = BitTracer(collapse = {'__init__', 'outgoing', 'step'})
    msg1 = k.format("Reify semantics as referentless embeddings", clip=True)
    b1 = tracer.run(f, m=msg1, k=k)
    msg2 = k.format("Test", clip=True)
    b2 = tracer.run(f, m=msg2, k=k)
    # tracer.mark_differences(b1, b2)
    visualize(b2)
