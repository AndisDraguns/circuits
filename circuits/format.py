from dataclasses import dataclass
from typing import Any, Literal, TypeGuard
from collections.abc import Callable, Iterator
import builtins

from circuits.core import Bit, const


@dataclass(frozen=True, eq=False, slots=True)
class Bits:
    """Represents a list of bits in various formats.
    For example, Bits(42).bitstr -> '101010'."""

    bit_tuple: tuple[Bit, ...]

    def __repr__(self):
        return f"Bits({self.bitstr})"

    def __init__(self, value: Any, min_length: int | None = None) -> None:
        bitlist = self._bitlist_from_value(value)
        if min_length is not None:
            padding_length = max(0, min_length - len(bitlist))
            bitlist = const("0" * padding_length) + bitlist
        object.__setattr__(self, "bit_tuple", tuple(bitlist))

    @staticmethod
    def _is_bit_list(lst: list[Any]) -> TypeGuard[list[Bit]]:
        return all(isinstance(el, Bit) for el in lst)

    @staticmethod
    def _is_bool_int_list(lst: list[Any]) -> TypeGuard[list[int | bool]]:
        return all(isinstance(el, (bool, int)) for el in lst)

    @classmethod
    def _bitlist_from_value(
        cls, value: int | bytes | str | list[Bit] | list[bool | int]
    ) -> list[Bit]:
        """Infer value type and convert it to list[Bit]."""
        if isinstance(value, cls):
            return value.bitlist
        match value:
            case int():
                n_bits = max(value.bit_length(), 1)
                return const(format(value, f"0{n_bits}b"))
            case bytes():  # Convert each byte into 8 bits, flatten the result:
                return [b for int8 in value for b in const(format(int8, "08b"))]
            case str():
                return cls.from_str(value).bitlist
            case list() if cls._is_bit_list(value):
                return value
            case list() if cls._is_bool_int_list(value):
                return const([int(v) for v in value])
            case _:
                # TODO: more informative error message, e.g. list of types received
                raise ValueError(f"Cannot create Bits from {type(value)}")

    @classmethod
    def from_str(
        cls, s: str, stype: Literal["bitstr", "hex", "text"] | None = None
    ) -> "Bits":
        """Convert string to Bits. If string type is not provided, infer it."""
        hex_chars = "0123456789abcdefABCDEF"
        if stype is None:
            if set(s) <= {"0", "1"}:
                stype = "bitstr"
            elif set(s) <= set(hex_chars) and len(s) % 2 == 0:
                stype = "hex"
            else:
                stype = "text"
        match stype:
            case "bitstr":
                return cls(const(s))
            case "hex":
                return cls(bytes.fromhex(s))
            case "text":
                return cls(s.encode("utf-8"))

    @property
    def bitlist(self) -> list[Bit]:
        return list(self.bit_tuple)

    @property
    def ints(self) -> list[int]:  # e.g. [1,0,1,0,1,0]
        return [int(b.activation) for b in self.bitlist]

    @property
    def bitstr(self) -> str:  # e.g. '101010'
        return "".join(map(str, self.ints))

    @property
    def int(self) -> int:  # e.g. 42
        return int(self.bitstr, 2)

    @property
    def bytes(self) -> bytes:
        if len(self) % 8:
            raise ValueError("Length must be multiple of 8 for bytes conversion")
        return bytes(int(self.bitstr[i : i + 8], 2) for i in range(0, len(self), 8))

    @property
    def hex(self) -> str:
        return self.bytes.hex()

    @property
    def text(self) -> str:
        """As text, replacing non-utf-8 characters with a replacement char"""
        return self.bytes.decode("utf-8", errors="replace")

    def __len__(self) -> builtins.int:
        return len(self.bitlist)

    def __iter__(self) -> Iterator[Bit]:
        return iter(self.bitlist)

    def __add__(self, other: "Bits") -> "Bits":
        return Bits(self.bitlist + other.bitlist)


def format_bits(message: Bits, bit_len: int = 1144) -> Bits:
    """Ensure that message has bit_len bits, by cropping / appending zeros"""
    m = message.bitstr[:bit_len]  # crop
    p = "0" * (bit_len - len(m))
    return Bits(m + p)


def format_msg(message: str, bit_len: int = 1144, pad: str = "_") -> Bits:
    """Ensure that message has bit_len bits, by appending pad symbols and cropping"""
    pad_len = bit_len - len(Bits(message))
    n_pad = 1 + pad_len // len(Bits(pad))
    bits = Bits(message + pad * n_pad)
    return format_bits(bits, bit_len)


def bitfun(function: Callable[..., Any]) -> Callable[..., Bits]:
    """Create a function with Bits instead of list[Bit] in inputs and output"""
    def bits_function(*args: Bits | Any, **kwargs: dict[str, Any]) -> Bits:
        modified_args = tuple(
            arg.bitlist if isinstance(arg, Bits) else arg for arg in args
        )
        return Bits(function(*modified_args, **kwargs))

    return bits_function


# def track(func: Callable[..., Any], prefix:str|None=None) -> Callable[..., Bits]:
#     """Decorator to track the name of the function that created the bits"""
#     def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Bits]:
#         name = func.__name__
#         if prefix:
#             name = f"{prefix}_{name}"
#         res = func(*args, **kwargs)
#         if isinstance(res, list):
#             for i, bit in enumerate(res):
#                 bit.metadata['name'] = f'{func.__name__}[{i}]'
#         else:
#             res.metadata['name'] = func.__name__
#         return res
#     return wrapper

# def track(func: Callable[..., list[Bit]]) -> Callable[..., list[Bit]]:
#     """Decorator to track the name of the function that created the bits"""
#     def wrapper(*args: Any, **kwargs: Any) -> list[Bit]:
#         name = func.__name__
#         res = func(*args, **kwargs)
#         if isinstance(res, Bit):
#             res.metadata['name'] = name
#         else:
#             try:
#                 resb = Bits(res).bitlist
#                 for i, bit in enumerate(resb):
#                     bit.metadata['name'] = f'{name}[{i}]'
#                 res = resb
#             except ValueError:
#                 # If res is not a Bits object, we can't track it
#                 pass
#         return res
#     return wrapper

# def track(func: Callable[..., list[Bit]]) -> Callable[..., list[Bit]]:
#     """Decorator to track the name of the function that created the bits"""
#     def wrapper(*args: Any, **kwargs: Any) -> list[Bit]:
#         res = func(*args, **kwargs)
#         for i, bit in enumerate(res):
#             bit.metadata['name'] = f'{func.__name__}[{i}]'
#         return res
#     return wrapper


# def track(func: Callable[..., list[Bit]], name: str|None=None) -> Callable[..., list[Bit]]:
#     """Decorator to track the name of the function that created the bits"""
#     def wrapper(*args: Any, **kwargs: Any) -> list[Bit]:
#         prefix = name if name else func.__name__
#         res = func(*args, **kwargs)
#         for i, bit in enumerate(res):
#             if 'name' in bit.metadata:
#                 prefix = bit.metadata['name'] + '.' + prefix
#             bit.metadata['name'] = f'{prefix}[{i}]'
#         return res
#     return wrapper


# def track(func: Callable[..., list[Bit]], name: str|None=None) -> Callable[..., list[Bit]]:
#     """Decorator to track the name of the function that created the bits"""
#     def wrapper(*args: Any, **kwargs: Any) -> list[Bit]:
#         prefix = name if name else func.__name__
#         res = func(*args, **kwargs)
#         for i, bit in enumerate(res):
#             if 'name' in bit.metadata:
#                 prefix = bit.metadata['name'] + '.' + prefix
#             bit.metadata['name'] = f'{prefix}[{i}]'
#         return res
#     return wrapper


# from typing import Any, TypeVar
# from collections.abc import Mapping, Iterable
# T = TypeVar('T')
# def track(func: Callable[..., T], name: str|None = None) -> Callable[..., T]:
#     """Decorator to track Bit objects at any nesting depth"""
#     def wrapper(*args: Any, **kwargs: Any) -> T:
#         prefix = name if name else func.__name__
#         result = func(*args, **kwargs)
        
#         def process(item: Any, path:str=""):
#             # Base case: found a Bit
#             if hasattr(item, "metadata"):
#                 old_name = item.metadata.get('name', '')
#                 full_path = f"{prefix}{path}{'.' + old_name if old_name else ''}"
#                 item.metadata['name'] = full_path
#                 return
                
#             # Skip non-container types and strings
#             if not isinstance(item, Iterable) or isinstance(item, str):
#                 return
                
#             # Handle mappings (dict-like)
#             if isinstance(item, Mapping):
#                 for k, v in item.items():  # type: ignore[reportUnknownMemberType]
#                     process(v, f"{path}[{k}]")
#                 return
                
#             # Handle sequences (list-like)
#             try:
#                 for i, v in enumerate(item):  # type: ignore[reportUnknownMemberType]
#                     process(v, f"{path}[{i}]")
#             except (TypeError, ValueError):
#                 pass  # Not indexable
        
#         process(result)
#         return result
        
#     return wrapper



# from typing import Any, TypeVar
# from functools import wraps
# from collections.abc import Mapping, Iterable

# T = TypeVar('T')

# def track(func: Callable[..., T]) -> Callable[..., T]:
#     """Decorator that properly stacks trackers for hierarchical naming"""
#     @wraps(func)
#     def wrapper(*args: Any, **kwargs: Any) -> T:
#         result = func(*args, **kwargs)
        
#         # Process the result to add or append metadata
#         def process(item: Any, path:str=""):
#             # Base case: found a Bit object
#             if hasattr(item, "metadata") and hasattr(item.metadata, "__setitem__"):
#                 current_path = f"{func.__name__}{path}"
#                 if 'name' in item.metadata:
#                     item.metadata['name'] = f"{current_path}.{item.metadata['name']}"
#                 else:
#                     item.metadata['name'] = current_path
#                 return
                
#             # Skip non-container types and strings
#             if not isinstance(item, Iterable) or isinstance(item, str):
#                 return
                
#             # Handle mappings (dict-like)
#             if isinstance(item, Mapping):
#                 for k, v in item.items():  # type: ignore[reportUnknownMemberType]
#                     process(v, f"[{k}]")
#                 return
                
#             # Handle sequences (list-like)
#             try:
#                 for i, v in enumerate(item):  # type: ignore[reportUnknownMemberType]
#                     process(v, f"[{i}]")
#             except (TypeError, ValueError):
#                 pass  # Not indexable
        
#         process(result)
#         return result
        
#     return wrapper