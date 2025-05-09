# Visualize
from io import BytesIO
from dataclasses import dataclass, field
import torch as t
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Literal, Any


@dataclass
class MatrixPlot:
    """Visualize a 2D matrix"""

    init_m: t.Tensor | list[list[int]] | list[list[float]]
    scale: float = 0.01
    clip_val: float | int = 20
    raster: bool = False
    quick: bool = False
    downsample_factor: int = 16
    downsample_kernel: Literal["mean", "median", "max_abs"] = "max_abs"
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
            return fig

    def draw(self) -> None:
        """Display a figure as raster or vector graphics"""
        fig = self.get_matrix_figure()
        if self.quick:
            fig.show()  # quick raster without transparency
            return
        buffer = BytesIO()
        format = "png" if self.raster else "svg"
        plt.savefig(buffer, format=format, transparent=True, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]
        buffer.seek(0)
        plt.close(fig)
        self.load_and_draw(buffer)

    def save(self, filename: str) -> None:
        fig = self.get_matrix_figure()  # type: ignore[reportUnknownMemberType]
        format = "png" if self.raster else "svg"
        plt.savefig(filename, format=format, transparent=True, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]

    def load_and_draw(self, file: str | BytesIO) -> None:
        try:
            from IPython.display import SVG, Image, display  # type: ignore[reportUnknownMemberType]

            if self.raster:
                data = file if isinstance(file, str) else file.getvalue()
                display(Image(data, retina=True))
            else:
                data = (
                    file if isinstance(file, str) else file.getvalue().decode("utf-8")
                )
                display(SVG(data))
        except:
            raise ValueError("load_and_draw is implemented only in IPython")


def draw(m: t.Tensor | list[list[int]] | list[list[float]], **kwargs: Any) -> None:
    """Visualize a 2D matrix"""
    with t.inference_mode():
        MatrixPlot(m, **kwargs).draw()


# Example:
# plot_kwargs = {'scale': 0.01, 'clip_val': 20, 'raster': False, 'quick': False,
#                 'downsample_factor': 16, 'downsample_kernel': 'max_abs'}
# draw(t.randn(1000, 1000), **plot_kwargs)
