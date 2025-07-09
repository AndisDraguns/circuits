from typing import Any
from collections.abc import Iterable

def process(item: Any, path:str="", prefix: str="") -> None:
    """Recursively process an object to add metadata to Bit objects."""
    if hasattr(item, "metadata"):  # Base case: found a Bit
        old_name = item.metadata.get('name', '')
        full_path = f"{prefix}{path}{'.' + old_name if old_name else ''}"
        item.metadata['name'] = full_path
        return
    # Skip non-container types and strings
    if not isinstance(item, Iterable) or isinstance(item, str):
        return
    try:
        for i, v in enumerate(item):  # type: ignore[reportUnknownMemberType]
            process(v, f"{path}{i}", prefix)
    except (TypeError, ValueError):
        pass  # Not indexable


def name(x: Any, name: str):
    """Name Bit variables in a given object."""
    process(x, prefix=name)


import inspect
def name_vars(exclude_args: bool = True) -> None:
    """Call this at the end of your function to name Bit vars in local variables."""
    frame = inspect.currentframe()
    if not frame or not frame.f_back:
        return
    try:
        local_vars = frame.f_back.f_locals
        if exclude_args:
            code = frame.f_back.f_code
            arg_names = code.co_varnames[:code.co_argcount]
            local_vars = {n: v for n, v in local_vars.items() if n not in arg_names}
        for n, v in local_vars.items():
            process(v, prefix=n)
    finally:
        del frame
