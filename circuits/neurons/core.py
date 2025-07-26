from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any


# import inspect
# def get_fn_stack(top_fn: str) -> str:
#     """Gets the function call stack up to top_fn"""
#     stack = inspect.stack()
#     names = [frame.function for frame in stack]
#     top_index = -1
#     if top_fn in names:
#         top_index = names.index(top_fn)
#     names = names[1: top_index]
#     names = filter(lambda n: n[:2] != "__", names)
#     excluded = ['const', 'gate', 'outgoing', '<lambda>']
#     names = list(filter(lambda n: n not in excluded, names))
#     n = '.'.join(reversed(names))
#     n = n.replace(".<listcomp>", "[i]")
#     return n

# Core MLP classes
@dataclass(frozen=True, eq=False, slots=True)
class Signal:
    """A connection point between neurons, with an activation value"""

    activation: bool | float
    source: "Neuron"
    metadata: dict[str, str] = field(default_factory=dict[str, str])
    trace: list[Any] = field(default_factory=list[Any])
    # def __post_init__(self):
    #     self.metadata['name'] = get_fn_stack('compiled')

    def __repr__(self):
        return f"Signal({self.activation})"


@dataclass(frozen=True, eq=False, slots=True)
class Neuron:
    incoming: tuple[Signal, ...]
    weights: tuple[float, ...] | tuple[int, ...]
    bias: float | int
    activation_function: Callable[[float | int], float | bool]

    @property
    def outgoing(self) -> Signal:  # creates new Signal
        summed = sum(v.activation * w for v, w in zip(self.incoming, self.weights))
        return Signal(self.activation_function(summed + self.bias), source=self)


# Linear threshold circuits
Bit = Signal
BitFn = Callable[[list[Bit]], list[Bit]]


def step(x: float | int) -> bool:
    return x >= 0


def gate(incoming: list[Bit], weights: list[int], threshold: int) -> Bit:
    """Create a linear threshold gate as a boolean neuron with a step function"""
    return Neuron(tuple(incoming), tuple(weights), -threshold, step).outgoing


def const(values: list[bool] | list[int] | str) -> list[Bit]:
    """Create constant list[Bit] from bits represented as bool, 0/1 or '0'/'1.
    Bits are negated because a threshold of 1 yields 0 and vice versa.'"""
    negated = [not bool(int(v)) for v in values]
    return [gate([], [], int(v)) for v in negated]


# Example:
# def and_(x: list[Bit]) -> Bit: return gate(x, [1]*len(x), len(x))
# and_(const('110'))  # Computes '1 and 1 and 0', which equals 0.
