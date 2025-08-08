from io import BytesIO
from dataclasses import dataclass, field
from typing import Literal, Any

import torch as t
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from IPython.display import HTML, SVG, Image, display  # type: ignore[reportUnknownMemberType]

from utils.misc import filter_kwargs


Matrix = t.Tensor | list[list[int]] | list[list[float]]


@dataclass
class MatrixPlot:
    """Visualize a 2D matrix"""

    init_m: t.Tensor
    size: float = 1
    square: bool = True
    fixed_height: bool = False
    fixed_width: bool = False
    fixed_pixel_size: bool = False
    pixel_scale: float = 0.001
    vector_factor: float | None = 0.1  # display vector as a fraction of the square size
    transpose_vectors: bool = False
    raster: bool = True
    quick: bool = False
    downsample_factor: int = 1
    downsample_kernel: Literal['mean', 'median', 'max_abs'] = 'max_abs'
    color_norm: Literal['linear', 'log'] = 'linear'
    clip: float | int | None = 1
    m: t.Tensor = field(default_factory=lambda: t.Tensor([[0, 0], [0, 0]]), init=False)


    def __post_init__(self) -> None:
        self.ensure_2D()
        self.downsample()
        if self.transpose_vectors and 1 in self.m.shape:
            self.m = self.m.transpose(0, 1)
        self.m = self.m.flip(0)  # flip to match the image coordinate system


    def ensure_2D(self) -> None:
        """Ensure that m is 2D"""
        m = t.Tensor(self.init_m).float()
        while len(m.size()) < 2:
            m = t.unsqueeze(m, -1)
        if m.ndim != 2:
            raise ValueError("Matrix must have 2 or fewer dimensions!")
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
        result = t.zeros((new_h, new_w)).float()
        for i in range(new_h):
            for j in range(new_w):
                block = self.m[i * k : min((i + 1) * k, h), j * k : min((j + 1) * k, w)]
                result[i, j] = self.kernel(block)
        self.m = result


    def get_figsize(self) -> tuple[float, float]:
        """Get the figure size in inches"""
        h, w = self.m.shape
        s = self.size
        if self.raster:
            s *= 2  # account for Image retina option in display()
        if self.square:
            width, height = s, s
            if self.vector_factor and h == 1:  # is row vector
                height *= self.vector_factor
            if self.vector_factor and w == 1:  # is column vector
                width *= self.vector_factor
            return width, height
        elif self.fixed_height:
            return s * w / h, s
        elif self.fixed_width:
            return s, s * h / w
        else:
            pixel_size = s * self.pixel_scale
            return pixel_size * w, pixel_size * h


    def get_color_norm(self) -> mcolors.Normalize:
        """Get the color normalization for the matrix"""
        c = self.clip
        if not c:
            c = t.max(t.abs(self.m)).item()
        if self.color_norm == "linear":
            return mcolors.Normalize(vmin=-c, vmax=c)
        elif self.color_norm == "log":
            return mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-c, vmax=c)
        else:
            raise ValueError(f"Unknown color norm: {self.color_norm}")


    def get_matrix_figure(self) -> Figure:
        """Create a plt figure of a 2D matrix"""
        h, w = self.m.shape
        fig, ax = plt.subplots()   # type: ignore[reportUnknownMemberType]
        fig.set_size_inches(self.get_figsize())
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        norm = self.get_color_norm()
        with t.inference_mode():
            X, Y = t.meshgrid(t.arange(w + 1), t.arange(h + 1), indexing="xy")
            X.float()  # for compatibility with matplotlib
            Y.float()
            pcm = ax.pcolormesh(X, Y, self.m, cmap="RdBu", norm=norm, edgecolors="face")  # type: ignore[reportUnknownMemberType]
            if not self.raster:  # 2x pcolormesh to prevent SVG seams:
                pcm = ax.pcolormesh(X, Y, self.m, cmap="RdBu", norm=norm, edgecolors="face")  # type: ignore[reportUnknownMemberType]
            pcm.set_edgecolor("none")  # removes the edgecolors='face' distortion
            ax.set_anchor("NW")
            ax.axis("off")
            return fig


    def get_buffer(self) -> BytesIO:
        """Get a buffer with the figure"""
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
            if isinstance(file, str):
                data = file
            else:
                data = file.getvalue().decode("utf-8")
            display(SVG(data))


def get_matrix_grid_html(matrices: list[t.Tensor], **kwargs: Any) -> str:
    "For displaying matrices in a flexible grid, using IPython.display.HTML"
    gap = kwargs.get("gap", 5)
    import base64
    css = f"<style>.matrix-container {{display: flex; flex-wrap: wrap; gap: {gap}px;}}</style>"
    matrices_html: list[str] = []
    with t.inference_mode():
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


def draw(m: t.Tensor, **kwargs: Any) -> None:
    """Visualize a 2D matrix"""
    with t.inference_mode():
        MatrixPlot(m, **kwargs).draw()


def plot(matrices: list[t.Tensor] | t.Tensor, **kwargs: Any) -> None:
    """Visualize matrices of 2 or fewer dimensions in an HTML grid"""
    if isinstance(matrices, t.Tensor):
        matrices = [matrices]
    html_str = get_matrix_grid_html(matrices, **kwargs)
    display(HTML(html_str))


# Example:
# m = t.randn(100, 20)
# plot(m)
