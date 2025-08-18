from dataclasses import dataclass, field
from turtle import tracer
from typing import NamedTuple

from circuits.neurons.core import Bit
from circuits.utils.ftrace import Tracer
from circuits.utils.format import Bits
from circuits.utils.blocks import Block

# TODO: add copies
# TODO: fix display of functions that do not create bits

@dataclass(frozen=True)
class Color:
    """HSL color representation (hue, saturation, lightness)"""
    h: float  # hue
    s: float  # saturation
    l: float  # lightness

    @property
    def css(self) -> str:
        return f"hsla({self.h}, {self.s}%, {self.l}%, 1.0)"

    def __add__(self, other: 'Color') -> 'Color':
        return Color((self.h+other.h)%360, min(self.s+other.s, 100), min(self.l+other.l, 100))


class Rect(NamedTuple):
    """Rectangle with percentage-based coordinates"""
    x: float
    y: float
    w: float
    h: float
    
    def shrink(self, amount: float) -> 'Rect':
        """Shrink rectangle by amount on all sides"""
        half = amount / 2
        return Rect(self.x+half, self.y+half, self.w-amount, self.h-amount)

    def to_percentages(self, root_w: float, root_h: float) -> 'Rect':
        """Convert absolute coordinates to percentages"""
        return Rect(self.x/root_w*100, self.y/root_h*100, self.w/root_w*100, self.h/root_h*100)


@dataclass
class VisualizationConfig:
    """Configuration for block visualization"""
    base_color: Color = field(default_factory=lambda: Color(180, 95, 90))
    hue_rotation: float = 2
    lightness_decay: float = 8
    highlight_transform: Color = field(default_factory=lambda: Color(200, 0, 0))
    non_live_transform: Color = field(default_factory=lambda: Color(90, 0, 0))
    hover_transform: Color = field(default_factory=lambda: Color(5, 0, -10))
    max_shrinkage: float = 0.95
    max_output_chars: float = 50

    def get_shrink_amount(self, depth: int, max_depth: int) -> float:
        """Calculate shrink amount for given depth"""
        return depth * self.max_shrinkage / (max_depth + 1)
    
    def get_color(self, depth: int, is_live: bool, highlight: bool) -> Color:
        """Calculate color for given depth"""
        color = self.base_color + Color(depth*self.hue_rotation, 0, -depth*self.lightness_decay)
        if not is_live:
            color = color + self.non_live_transform
        if highlight:
            color = color + self.highlight_transform
        return color


def generate_block_html(node: Block, config: VisualizationConfig, 
                        max_depth: int, root_dims: tuple[float, float]) -> str:
    """Generate HTML for a single block and its children"""
    if node.name in {'__init__', 'outgoing'}:
        return ""

    # Create rectangle and apply transformations
    rect = Rect(node.x, node.y, node.w, node.h)
    rect = rect.shrink(config.get_shrink_amount(node.depth, max_depth))
    rect = rect.to_percentages(*root_dims)
    small = False
    if rect.w <= 0 or rect.h <= 0:
        # return ""  # exclude invisible elements
        new_w = 0.2 if rect.w <= 0 else rect.h
        new_h = 0.2 if rect.h <= 0 else rect.w
        rect = Rect(rect.x, rect.y, new_w, new_h)
        small = True
    
    # Get color
    color = config.get_color(node.depth, node.is_live, node.highlight)
    hover_color = color + config.hover_transform

    if small:
        color += Color(50, 0, -100)
    if node.name == 'copy':
        color += Color(-50, 0, 0)

    # Generate tooltip
    out_str = Bits(list(node.outputs)).bitstr
    truncated = out_str[:config.max_output_chars]
    if len(out_str) > config.max_output_chars:
        truncated += '...'
    tooltip = node.full_info()

    # Generate children HTML
    children_html = ''.join(
        generate_block_html(child, config, max_depth, root_dims) 
        for child in node.children
    )
    
    return f'''
    <div class="block" 
         title="{tooltip}"
         style="--x:{rect.x}; --y:{rect.y}; --w:{rect.w}; --h:{rect.h}; 
                --color:{color.css}; --hover-color:{hover_color.css};">
        {children_html}
    </div>'''


HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Call Tree Block Visualization</title>
<style>
    body {{
        margin: 0;
        font-family: system-ui, -apple-system, sans-serif;
        background: #1a1a1a;
        color: #f0f0f0;
    }}
    
    .vis-container {{
        position: relative;
        width: 100vw;
        height: 100vh;
    }}
    
    .block {{
        position: fixed;
        box-sizing: border-box;
        left: calc(var(--x) * 1vw);
        bottom: calc(var(--y) * 1vh);
        width: calc(var(--w) * 1vw);
        height: calc(var(--h) * 1vh);
        background-color: var(--color);
        cursor: pointer;
    }}

    /* highlight a block but not its parents */
    .block:hover:not(:has(.block:hover)) {{
        background-color: var(--hover-color);
    }}
    
    .block.collapsed > .block {{
        display: none;
    }}
    
    #info {{
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(40, 40, 40, 0.80);
        border: 1px solid #555;
        padding: 15px;
        border-radius: 8px;
        max-width: 400px;
        font: 13px 'Courier New', monospace;
        white-space: pre-wrap;
        cursor: pointer;
    }}
    #info.icon {{
        width: 20px;
        height: 20px;
        overflow: hidden;
        font-size: 0;
    }}
    #info.icon::before {{
        content: "i";
        font-size: 20px;
        text-align: center;
    }}

</style>
</head>
<body>
    <div class="vis-container">{blocks}</div>
    <div id="info"></div>
    <script>
        const info = document.getElementById('info');
        info.textContent = 'Click a block to display its info';
        info.classList.add('icon'); 
        document.querySelectorAll('.block').forEach(block => {{
            block.addEventListener('click', e => {{
                e.stopPropagation();
                if (e.detail === 1) {{
                    info.textContent = e.currentTarget.title;
                    info.classList.remove('icon');  // expand info panel
                }}
            }});
            block.addEventListener('dblclick', e => {{
                e.stopPropagation();
                e.currentTarget.classList.toggle('collapsed');
            }});
        }});

        // Info panel click - toggle icon/expanded state
        info.addEventListener('click', e => {{
            e.stopPropagation();
            info.classList.toggle('icon');
        }});
    </script>
</body>
</html>'''


def save_visualization(root: Block,
                      filename: str = "index.html",
                      config: VisualizationConfig | None = None) -> None:
    """Generate and save visualization to file"""
    config = config or VisualizationConfig()
    b = root.children[0]  # get rid of the wrapper
    b.parent = None
    assert b.w > 0 and b.h > 0 
    blocks_html = generate_block_html(b, config, b.max_leaf_depth, (b.w, b.h))
    html = HTML_TEMPLATE.format(blocks=blocks_html)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)


# Example usage
if __name__ == '__main__':
    # tracer = Tracer(use_defaults=True)
    tracer = Tracer(Bit)
    from circuits.examples.keccak import Keccak
    from circuits.neurons.core import Bit
    def f(m: Bits, k: Keccak) -> list[Bit]:
        return k.digest(m).bitlist
    k = Keccak(c=10, l=0, n=1, pad_char='_')

    msg1 = k.format("Reify semantics as referentless embeddings", clip=True)
    b1 = Block.from_node(tracer.run(f, m=msg1, k=k)).process()

    msg2 = k.format("Test", clip=True)
    b2 = Block.from_node(tracer.run(f, m=msg2, k=k)).process()

    b2.highlight_differences(b1)

    save_visualization(b2)
    # save_visualization(b1)



# from circuits.utils.format import Bits
# # from circuits.sparse.compile import compiled_from_io
# def test_subset_parity_sandbagging():
#     """Test sandbagging with a subset parity circuit.
#     Without the trigger, the circuit should rarely work"""
    
#     # Build the sandbagging circuit
#     inp_len = 8
#     trigger = const('11010011')
#     # trigger = const('00000000')
#     k = Keccak(c=30, l=2, n=1, auto_c=True)
#     k.d = k.b
#     subset_indices = [0,2,4,6]  # fixed for testing
#     subset_parity = get_subset_parity(inp_len, subset_indices)
#     sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)




# if __name__ == '__main__':
#     tracer = Tracer()

#     from circuits.neurons.core import const
#     from circuits.examples.keccak import Keccak
#     from circuits.tests.backdoors_test import get_subset_parity, get_sandbagger, gen_random_bitlist
#     # clone, is_parity
#     inp_len = 5
#     trigger = const('11010')
#     k = Keccak(c=16, l=0, n=1, auto_c=True)
#     k.d = k.b
#     subset_indices = [0,2,4]  # fixed for testing
#     subset_parity = get_subset_parity(inp_len, subset_indices)
#     sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

#     def test_sandbag(t: list[Bit], x: list[Bit]) -> list[Bit]:
#         result = sandbagger(t, x)
#         # assert is_parity(x, result, subset_indices)
#         return result
        
#     x1 = gen_random_bitlist(inp_len)
#     trace = tracer.run(test_sandbag, t=trigger, x=x1)

#     x2 = gen_random_bitlist(inp_len)
#     trace2 = tracer.run(test_sandbag, t=trigger, x=x2)
#     # t2 = gen_random_bitlist(inp_len)
#     # trace2 = tracer.run(test_sandbag, t=t2, x=x1)

#     trace2.highlight_differences(trace)
#     save_visualization(trace2)