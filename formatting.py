from typing import Any, Literal
from collections.abc import Callable, Iterator
# import torch as t
import builtins

from core import *


from typing import TypeGuard
def is_bit_list(value: list[Any]) -> TypeGuard[list[Bit]]:
    return all(isinstance(x, Bit) for x in value)
def is_bool_int_list(value: list[Any]) -> TypeGuard[list[int | bool]]:
    return all(isinstance(x, (bool, int)) for x in value)

@dataclass(frozen=True, eq=False)
class Bits:
    """Represents a list of bits in various formats.
    For example, Bits(42).bitstr -> '101010'."""
    bitlist: list[Bit]
    def __repr__(self): return f"Bits({self.bitstr})"

    def __init__(self, value: Any, min_length: int | None = None) -> None:
        bitlist = self._bitlist_from_value(value)
        if min_length is not None:
            padding_length = max(0, min_length - len(bitlist))
            bitlist = const('0' * padding_length) + bitlist
        object.__setattr__(self, 'bitlist', bitlist)


    @classmethod
    def _bitlist_from_value(cls, value: int | bytes | str | list[Bit] | list[bool | int]
                            ) -> list[Bit]:
        """Infer value type and convert it to list[Bit]."""
        if isinstance(value, cls):
            return value.bitlist
        match value:
            case int():
                n_bits = max(value.bit_length(), 1)
                return const(format(value, f'0{n_bits}b'))
            case bytes():  # Convert each byte into 8 bits, flatten the result:
                return [b for int8 in value for b in const(format(int8, '08b'))]
            case str():
                return cls.from_str(value).bitlist
            case list() if is_bit_list(value):
                return value
            case list() if is_bool_int_list(value):
                return const([int(v) for v in value])
            # case _ if hasattr(value, 'tolist') and isinstance(value.tolist(), list):
            #     return const(value.tolist())  # Handle tensor-like objects
            case _:
                raise ValueError(f"Cannot create Bits from {type(value)}")

    @classmethod
    def from_str(cls, s: str, stype: Literal['bitstr', 'hex', 'text'] | None = None) -> 'Bits':
        """Convert string to Bits. If string type is not provided, infer it."""
        hex_chars = '0123456789abcdefABCDEF'
        if stype is None:
            if set(s) <= {'0', '1'}:
                stype = 'bitstr'
            elif set(s) <= set(hex_chars) and len(s) % 2 == 0:
                stype = 'hex'
            else:
                stype = 'text'
        match stype:
            case 'bitstr': return cls(const(s))
            case 'hex': return cls(bytes.fromhex(s))
            case 'text': return cls(s.encode('utf-8'))
            case _: raise ValueError(f"Unknown string type: {stype}")

    @property
    def ints(self) -> list[int]:  # e.g. [1,0,1,0,10]
        return [int(b.activation) for b in self.bitlist]

    @property
    def bitstr(self) -> str:  # e.g. '101010'
        return ''.join(map(str, self.ints))

    @property
    def int(self) -> int:  # e.g. 42
        return int(self.bitstr, 2)

    @property
    def bytes(self) -> bytes:
        if len(self) % 8:
            raise ValueError("Length must be multiple of 8 for bytes conversion")
        return bytes(int(self.bitstr[i:i+8], 2) for i in range(0, len(self), 8))

    @property
    def hex(self) -> str:
        return self.bytes.hex()

    @property
    def text(self) -> str:
        """As text, replacing non-utf-8 characters with a replacement char"""
        return self.bytes.decode('utf-8', errors='replace')

    # @property
    # def tensor(self) -> t.Tensor:
    #     return t.tensor(self.ints)

    def __len__(self) -> builtins.int:
        return len(self.bitlist)

    def __getitem__(self, idx: builtins.int | slice) -> 'Bit | Bits':
        if isinstance(idx, builtins.int):
            return self.bitlist[idx]
        return Bits(self.bitlist[idx])

    def __iter__(self) -> Iterator[Bit]:
        return iter(self.bitlist)

    def __add__(self, other: 'Bits') -> 'Bits':
        return Bits(self.bitlist + other.bitlist)


def format_msg(message: str, bit_len: int = 1144) -> Bits:
    """Append a message string with with '_' padding symbols, convert to Bits"""
    padded = message + '_'*(bit_len//8 - len(message))
    return Bits(padded)


def bitfun(function: Callable[[Any], Any]) -> Callable[..., Bits]:
    """Create a function with Bits instead of list[Bit] in inputs and output"""
    def bits_function(*args: Bits | Any, **kwargs: dict[str, Any]) -> Bits:
        modified_args = tuple(arg.bitlist if isinstance(arg, Bits) else arg for arg in args)
        return Bits(function(*modified_args, **kwargs))
    return bits_function


# # Example:
# print(Bits(42).bitstr)  # 42 in binary = 101010