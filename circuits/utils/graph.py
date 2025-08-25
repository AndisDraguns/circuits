from dataclasses import dataclass


@dataclass(frozen=True)
class Parent:
    index: int
    weight: int


@dataclass(frozen=True)
class Origin:
    """Connection info"""
    index: int
    incoming: tuple[Parent, ...]
    bias: int


@dataclass(frozen=True)
class Level:
    origins: tuple[Origin, ...]


@dataclass(frozen=True)
class Graph:
    levels: tuple[Level, ...]

    @property
    def shapes(self) -> tuple[tuple[int, int], ...]:
        widths = [len(level.origins) for level in self.levels]
        return tuple([(out_w, inp_w) for out_w, inp_w in zip(widths[1:], widths[:-1])])