from circuits.neurons.core import const
from circuits.neurons.operations import xors
from circuits.utils.format import Bits
from circuits.sparse.compile import compiled_from_io


def test_xors():
    a = const("101")
    b = const("110")
    f_res = xors([a, b])
    print(Bits(f_res))
    graph = compiled_from_io(a + b, f_res)
    g_res = graph.run(a + b)
    print(Bits(g_res))
    correct = [bool(ai.activation) ^ bool(bi.activation) for ai, bi in zip(a, b)]
    print(Bits(correct))


test_xors()
