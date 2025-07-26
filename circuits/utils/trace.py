

from circuits.neurons.core import Signal


import sys
from collections.abc import Callable
from typing import Any

# class Signal:
#     def __init__(self):
#         self.trace = [Call('', '')]


class Call:
    """
    Represents a single call in the call stack.
    E.g. indices = [[0], [[1,2]]] mean that signal was found in the fn call return twice:
    once at index [0], once at index [1,2].
    """
    def __init__(self, fname: str, call_nr: str):
        self.fname: str = fname
        self.call_number: str = call_nr
        self.subcalls: dict[str, dict[str, Call]] = {}  # subcalls[fname][call_nr] = subcall
        self.returns: list[list[int]] = []  # returns[return_nr][dimension] = index
        self.is_root = False
        self.root_name = ""
        if self.call_number == "":
            self.is_root = True

    def name_str(self) -> str:
        return self.root_name if self.is_root else f"{self.fname}-{self.call_number}"

    def returns_str(self) -> str:
        if len(self.returns) == 0:
            return ""
        elif len(self.returns) == 1:
            returns = "".join([f"[{str(i)}]" for i in self.returns[0]])
            if self.returns[0] == []:
                returns = "[]"
            return returns
        else:
            return '+'.join([str(r) for r in self.returns])

    def subcalls_str(self) -> str:
        subcalls_flat = self.subcalls_flat()
        match self.n_children():
            case 0:
                res = ""
            case 1:
                res = str(subcalls_flat[0])
            case _:
                res = '\n\t'.join(str(s) for s in subcalls_flat)
                res = '\n\t' + res
        return res

    def subcalls_flat(self) -> list['Call']:
        subcalls_unpacked = [list(s.values()) for s in list(self.subcalls.values())]
        return [x for xs in subcalls_unpacked for x in xs]

    def n_children(self) -> int:
        """Returns number of subcalls"""
        n = sum(len(subcall) for subcall in self.subcalls.values())
        return n
    
    def has_width(self) -> bool:
        """Returns True if any node in the tree has degree >= 2"""
        match self.n_children():
            case 0:
                return False
            case 1:
                return any(subcall.has_width() for subcall in self.subcalls_flat())
            case _:
                return True

    def __str__(self):
        subcalls_str = self.subcalls_str()
        separator = '' if self.n_children()==0 or self.is_root else '.'
        call_string = f"{self.name_str()}{self.returns_str()}{separator}{subcalls_str}"
        # if self.is_root:
        #     call_string = call_string[1:]  # remove leading dot for root call
        if self.has_width() and self.is_root:
            call_string = "\n" + call_string
        return call_string

    def __repr__(self):
        return self.__str__()


def filter_path(path: str, ignored: set[str] = set(), renamed: dict[str, str] = {}) -> str:
    """Removes ignored functions from a path"""
    pathlist = path.split('.')
    new_pathlist: list[str] = []
    if ']' not in pathlist[-1]:
        pathlist[-1] += '[]'  # emphasize that this path returns with '[]'
    for el in pathlist:
        fname = el.split('-')[0]
        if fname not in ignored:
            fname = renamed.get(fname, fname)  # rename if needed
            new_pathlist.append(f"{fname}-{el.split('-')[1]}")
    new_path = '.'.join(new_pathlist)
    return new_path


def add_path(path: str, root_call: Call):
    """Updates a root Call instance, including path in its trace tree"""
    calls_string = path.split('.')
    if len(path) == 0:
        return
    current_call: Call = root_call
    for subcall_str in calls_string:
        fname, suffix = subcall_str.split('-')
        call_nr = suffix.split('[')[0]

        # if no fname in subcalls
        if fname not in current_call.subcalls:
            subcall = Call(fname, call_nr)
            current_call.subcalls[fname] = {call_nr: subcall}
        
        # if no fname[call_nr] in subcalls[fname]
        elif call_nr not in current_call.subcalls[fname]:
            subcall = Call(fname, call_nr)
            current_call.subcalls[fname][call_nr] = subcall

        subcall = current_call.subcalls[fname][call_nr]
        if '[' in suffix:
            if suffix[-2]!='[':
                return_indices = subcall_str.split('[')[1:]
                return_indices = [int(idx[:-1]) for idx in return_indices]  # crop ']'
                if return_indices not in subcall.returns:
                    subcall.returns.append(return_indices)
            else:
                subcall.returns.append([])
                pass

        current_call = subcall


def trace(func: Callable[[Any], Any], ignored=set[str], renamed=dict[str, str], *args, **kwargs) -> Any:
    call_stack = []
    execution_counters = {}
    original_trace_func = sys.gettrace() # Save the original tracer

    def set_trace_func(trace_func: Callable[[Any], Any]):
        """Sets trace function using PyDev API. sys.settrace produces warnings."""
        if 'pydevd' in sys.modules:
            from pydevd_tracing import SetTrace
            SetTrace(trace_func)
        else:
            sys.settrace(trace_func)


    def process_return_value(value, base_path, signals_with_paths):
        def traverse(obj, path_suffix=''):
            current_path = base_path + path_suffix
            if isinstance(obj, Signal):
                if len(obj.trace) == 0:
                    obj.trace.append(Call('', ''))
                signals_with_paths.append((obj, current_path))
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    traverse(item, path_suffix + f'[{i}]')
        traverse(value)

    def trace_handler(frame, event, arg):
        # Skip built-ins and stdlib immediately
        filename = frame.f_code.co_filename
        # if filename.split('/')[-1] == 'core.py':
        #     return None
        if filename.startswith('<') or '/lib/python' in filename:
            return None  # Stop tracing this call tree

        fname = frame.f_code.co_name
        if fname in {'gate'}:
            return None

        try:
            if event == 'call':
                func_name = frame.f_code.co_name
                parent_full_path = '.'.join(f"{entry['name']}-{entry['count']}" for entry in call_stack)
                
                key = (parent_full_path, func_name)
                execution_counters[key] = execution_counters.get(key, -1) + 1
                
                call_entry = {
                    'name': func_name,
                    'count': execution_counters[key]
                }
                call_stack.append(call_entry)

            elif event == 'return':
                if call_stack:
                    full_path = '.'.join(f"{entry['name']}-{entry['count']}" for entry in call_stack)
                    
                    signals_with_paths = []
                    process_return_value(arg, full_path, signals_with_paths)
                    for signal, path in signals_with_paths:
                        filtered_path = filter_path(path, ignored, renamed)
                        if filtered_path is not None:
                            add_path(filtered_path, signal.trace[0])
                    
                    call_stack.pop()

        except Exception as e:
            print(f"Exception in trace handler: {e}")
            assert False, "Trace handler encountered an error"
        
        if original_trace_func:  # Call the original tracer if it exists
            original_trace_func(frame, event, arg)
        return trace_handler


    try:
        set_trace_func(trace_handler)  # Set the new tracer
        result = func(*args, **kwargs)
    finally:
        set_trace_func(original_trace_func)  # Restore the original tracer
    
    return result



def get_trace(node) -> dict[str, str | bool]:
    tr = node.original_signal.trace
    if len(tr)>0:
        return {'trace': str(tr[0]), 'found': True, 'has_width': tr[0].has_width()}
    else:
        return {'trace': "X", 'found': False, 'has_width': False}

def print_traces(graph):
    for i, layer in enumerate(graph.layers):
        # if i!= 5: continue
        # layer = layer[:2]
        print("\n" + f"Layer {i}:")
        layer_tr = [get_trace(node) for node in layer]
        layer_tr_new = []
        for j, tr in enumerate(layer_tr[:-1]):
            layer_tr_new.append(f"{tr['trace']}")
            if tr['has_width'] and not layer_tr[j+1]['has_width']:
                layer_tr_new.append("\n")
        layer_tr_new.append(f"{layer_tr[-1]['trace']}")
        layer_str = ", ".join(str(tr) for tr in layer_tr_new)
        if layer_str[0] == '\n':
            layer_str = layer_str[1:]
        print(layer_str)

# from circuits.sparse.compile import compiled_from_io
from circuits.examples.keccak import Keccak
# from circuits.utils.format import Bits
from circuits.neurons.core import Bit
def test() -> tuple[list[Bit], list[Bit]]:
    k = Keccak(c=20, l=1, n=2, pad_char='_')
    phrase = "Reify semantics as referentless embeddings"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)
    return message.bitlist, hashed.bitlist

ignored = ["<listcomp>", '<lambda>', "__init__"] + ["gate", "outgoing", "bitlist", "_bitlist_from_value", "test"]
ignored += ['from_str', 'bitlist_to_msg', 'digest', 'hash_state', 'crop_digest']
ignored += ['lanes_to_state', 'state_to_lanes', 'reverse_bytes', 'format', 'msg_to_state', 'rot', 'rho_pi', 'copy_lanes']
ignored = set(ignored)
renamed = {'const':'c', 'and_': 'and', 'or_': 'or'}
renamed.update({'not_': 'not'})

# graph = test()
# inp, out = test()
inp, out = trace(test, ignored, renamed)
from circuits.sparse.compile import compiled_from_io
graph = compiled_from_io(inp, out)
print_traces(graph)






# from circuits.neurons.core import const
# from circuits.neurons.operations import xors, ands
# from circuits.sparse.compile import compiled_from_io
# def test():
#     a = const('110')
#     b = const('101')
#     c = const('111')
#     res1 = xors([a, b])
#     res2 = xors([b, c])
#     res3 = ands([res1, res2])
#     graph = compiled_from_io(a + b + c, res3)
#     # print(graph)
#     return graph

# from circuits.sparse.compile import compiled_from_io
# from circuits.utils.format import Bits
# from circuits.neurons.operations import add
# def test():
#     a = 42
#     b = 39
#     a = Bits(a, 10).bitlist  # as Bits with 10 bits
#     b = Bits(b, 10).bitlist  # as Bits with 10 bits
#     result = add(a, b)  # as Bits with 10 bits
#     graph = compiled_from_io(a + b, result)
#     # print(graph)
#     return graph

# return "\n\n".join(
#     f"Layer {i}: " + ", ".join(f"{node.metadata.get('name', 'N')}" for node in layer)
#     for i, layer in enumerate(self.layers)
# )
