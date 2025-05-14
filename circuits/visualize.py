# Visualize
from io import BytesIO
from dataclasses import dataclass, field, fields
import torch as t
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Literal, Any
from IPython.display import HTML, SVG, Image, display  # type: ignore[reportUnknownMemberType]

Matrix = t.Tensor | list[list[int]] | list[list[float]]


@dataclass
class MatrixPlot:
    """Visualize a 2D matrix"""

    init_m: Matrix
    scale: float = 0.001
    clip_val: float | int = 20
    raster: bool = False
    quick: bool = False
    downsample_factor: int = 1
    downsample_kernel: Literal["mean", "median", "max_abs"] = "max_abs"
    square_size: int | None = None
    squish_biases: bool = True
    squish_biases_factor: int = 20
    m: t.Tensor = field(default_factory=lambda: t.Tensor([[0, 0], [0, 0]]), init=False)

    def __post_init__(self) -> None:
        self.ensure_2D()
        self.downsample()

    def ensure_2D(self) -> None:
        """Ensure that m is 2D"""
        m = t.Tensor(self.init_m)
        while len(m.size()) < 2:
            m = t.unsqueeze(m, -1)
        if m.ndim != 2:
            raise ValueError("m ndim must be <= 2")
        self.m = m

    def kernel(self, block: t.Tensor) -> t.Tensor:
        match self.downsample_kernel:
            case "mean":
                return t.mean(block)
            case "median":
                return t.median(block)
            case "max_abs":
                sup, inf = t.max(block), t.min(block)
                return sup if abs(sup) > abs(inf) else inf

    def downsample(self) -> None:
        """Downsample a 2D matrix by factor k"""
        h, w = self.m.shape
        k = self.downsample_factor
        new_h = (h + k - 1) // k
        new_w = (w + k - 1) // k
        result = t.zeros((new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                block = self.m[i * k : min((i + 1) * k, h), j * k : min((j + 1) * k, w)]
                result[i, j] = self.kernel(block)
        self.m = result

    def get_matrix_figure(self) -> Figure:
        """Create a plt figure of a 2D matrix"""
        h, w = self.m.shape
        s = self.scale
        if self.raster:
            s *= 2  # account for Image retina display option

        fig, ax = plt.subplots(figsize=(w * s, h * s))  # type: ignore[reportUnknownMemberType]
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        norm = mcolors.SymLogNorm(
            linthresh=0.1, linscale=0.1, vmin=-self.clip_val, vmax=self.clip_val
        )
        with t.inference_mode():
            X, Y = t.meshgrid(t.arange(w + 1), t.arange(h + 1), indexing="xy")
            # 2x to prevent seams. 'none' removes the edgecolors='face' distortion
            pcm = ax.pcolormesh(X, Y, self.m, cmap="RdBu", norm=norm, edgecolors="face")  # type: ignore[reportUnknownMemberType]
            pcm = ax.pcolormesh(X, Y, self.m, cmap="RdBu", norm=norm, edgecolors="face")  # type: ignore[reportUnknownMemberType]
            pcm.set_edgecolor("none")  # removes the edgecolors='face' distortion
            ax.set_anchor("NW")
            ax.axis("off")
            if self.square_size:
                width = self.square_size
                if self.squish_biases and len(self.m[0]) == 1:  # is bias
                    width = self.square_size / self.squish_biases_factor
                fig.set_size_inches(width, self.square_size)
            return fig

    def get_buffer(self) -> BytesIO:
        fig = self.get_matrix_figure()
        buffer = BytesIO()
        format = "png" if self.raster else "svg"
        plt.savefig(buffer, format=format, transparent=True, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]
        buffer.seek(0)
        plt.close(fig)
        return buffer

    def draw(self) -> None:
        """Display a figure as raster or vector graphics"""
        if self.quick:  # opaque raster
            fig = self.get_matrix_figure()
            fig.show()
            return
        else:
            buffer = self.get_buffer()
            self.load_and_draw(buffer)

    def save(self, filename: str) -> None:
        fig = self.get_matrix_figure()  # type: ignore[reportUnknownMemberType]
        format = "png" if self.raster else "svg"
        plt.savefig(filename, format=format, transparent=True, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]

    def load_and_draw(self, file: str | BytesIO) -> None:
        "Load and draw in IPython"
        if self.raster:
            data = file if isinstance(file, str) else file.getvalue()
            display(Image(data, retina=True))
        else:
            data = (
                file if isinstance(file, str) else file.getvalue().decode("utf-8")
            )
            display(SVG(data))


def filter_kwargs(cls: type, **kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in {f.name for f in fields(cls)}}


def get_matrix_grid_html(matrices: list[Matrix], **kwargs: Any) -> str:
    "For displaying matrices in a flexible grid. For use with IPython.display.HTML"
    gap = kwargs.get("gap", 5)
    import base64
    css = f"<style>.matrix-container {{display: flex; flex-wrap: wrap; gap: {gap}px;}}</style>"
    matrices_html: list[str] = []
    for m in matrices:
        filtered_kwargs = filter_kwargs(MatrixPlot, **kwargs)
        mplot = MatrixPlot(m, **filtered_kwargs)
        buffer_val = mplot.get_buffer().getvalue()
        if mplot.raster:
            img_str = base64.b64encode(buffer_val).decode('utf-8')
            html_str = f'<img src="data:image/png;base64,{img_str}" />'
        else:
            html_str = buffer_val.decode('utf-8')
        matrices_html.append(html_str)
    html = f"{css}<div class='matrix-container'>{''.join(matrices_html)}</div>"
    return html


def draw(m: t.Tensor | list[list[int]] | list[list[float]], **kwargs: Any) -> None:
    """Visualize a 2D matrix"""
    with t.inference_mode():
        MatrixPlot(m, **kwargs).draw()


def plot(matrices: list[Matrix], **kwargs: Any) -> HTML:
    return HTML(get_matrix_grid_html(matrices, **kwargs))


# # Example:
# plot_kwargs: dict[str, Any] = {'scale': 0.1, 'clip_val': 20, 'raster': True, 'quick': False,
#                 'downsample_factor': 16, 'downsample_kernel': 'max_abs', 'square_size': 10}
# m = t.randn(20, 10)
# # draw(m, **plot_kwargs)
# # MatrixPlot(m, **plot_kwargs).save('test.png')
# plot([m], gap=3, **plot_kwargs)




# from circuits.compile import compile_from_example
# from circuits.torch_mlp import StepMLP
# from circuits.format import bitfun
# from circuits.format import format_msg
# from circuits.examples.keccak import keccak, KeccakParams
# p = KeccakParams(c=20, l=3, n=3)
# print("params:", p)
# test_phrase = "Rachmaninoff"
# message = format_msg(test_phrase, bit_len=p.msg_len)
# hashed = bitfun(keccak)(message, c=p.c, l=p.l, n=p.n)
# layered_graph = compile_from_example(message.bitlist, hashed.bitlist)
# mlp = StepMLP.from_graph(layered_graph)
# print("layer sizes:", mlp.sizes)

# plot_kwargs = {'scale': 0.1, 'clip_val': 20, 'raster': True, 'quick': False,
#                 'downsample_factor': 1, 'downsample_kernel': 'max_abs', 'square_size': 10}
# layer_nr = -10
# matrix = mlp.net[layer_nr].weight
# # MatrixPlot(matrix, **plot_kwargs).save('test.png')

# print(layered_graph.layers[-9])



# from circuits.track import name_vars


# from circuits.compile import compile_from_dummy_io
# from circuits.torch_mlp import StepMLP
# from circuits.format import bitfun
# from circuits.core import Bit, const
# from circuits.format import Bits
# from circuits.operations import add, or_, xor, and_

# from circuits.core import gate
# def xor(x: list[Bit]) -> Bit:
#     m = [gate(x, [1] * len(x), i + 1) for i in range(len(x))]
#     name_vars()
#     return gate(m, [(-1) ** i for i in range(len(x))], 1)

# def add(a: list[Bit], b: list[Bit]) -> list[Bit]:
#     """Adds two integers in binary using a parallel adder.
#     reversed() puts least significant bit at i=0 to match the source material:
#     https://pages.cs.wisc.edu/~jyc/02-810notes/lecture13.pdf page 1."""
#     a, b = list(reversed(a)), list(reversed(b))
#     n = len(a)
#     p = [or_([a[i], b[i]]) for i in range(len(a))]
#     q = [[and_([a[i], b[i]] + p[i + 1 : k]) for i in range(k)] for k in range(n)]
#     c = const([0]) + [or_(q[k]) for k in range(1, n)]
#     s = [xor([a[k], b[k], c[k]]) for k in range(n)]
#     name_vars(False)
#     return list(reversed(s))

# bitlen = 5
# a = 1
# b = 2
# a = Bits(a, bitlen)
# b = Bits(b, bitlen)
# result = bitfun(add)(a, b)
# layered_graph = compile_from_dummy_io(a.bitlist+b.bitlist, result.bitlist)

# # i1 = const('110')
# # i2 = const('100')
# # from circuits.operations import bitwise
# # xors = bitwise(xor)
# # result = xors([i1,i2])
# # layered_graph = compile_from_example(i1+i2, result)

# mlp = StepMLP.from_graph(layered_graph)
# print(mlp.layer_stats)
# print(layered_graph)
