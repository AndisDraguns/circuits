from dataclasses import dataclass, field
from turtle import tracer
from typing import NamedTuple

from circuits.neurons.core import Bit
from circuits.utils.ftrace import CallNode, Tracer, Trace
from circuits.utils.format import Bits


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
    non_live_transform: Color = field(default_factory=lambda: Color(0, -40, 0))
    max_shrinkage: float = 1.4
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


def generate_block_html(node: CallNode, config: VisualizationConfig, 
                        max_depth: int, root_dims: tuple[float, float]) -> str:
    """Generate HTML for a single block and its children"""
    # Create rectangle and apply transformations
    rect = Rect(node.x, node.y, node.w, node.h)
    rect = rect.shrink(config.get_shrink_amount(node.depth, max_depth))
    rect = rect.to_percentages(*root_dims)
    if rect.w <= 0 or rect.h <= 0:
        return ""  # exclude invisible elements
    
    # Generate tooltip
    out_str = Bits([s for s, _ in node.outputs]).bitstr
    truncated = out_str[:config.max_output_chars]
    if len(out_str) > config.max_output_chars:
        truncated += '...'
    # tooltip = f"{node.info_str()}"
    # tooltip += f", d={node.depth}, out={truncated}, live={node.is_live}, highlight={node.highlight}"
    tooltip = node.full_info()
    
    # Get color
    color = config.get_color(node.depth, node.is_live, node.highlight)
    hover_color = color + Color(5, 0, -10)
    
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
        background: rgba(40, 40, 40, 0.95);
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


def save_visualization(trace: Trace,
                      filename: str = "index.html",
                      config: VisualizationConfig | None = None) -> None:
    """Generate and save visualization to file"""
    config = config or VisualizationConfig()
    r = trace.root
    blocks_html = generate_block_html(r, config, trace.max_depth, (r.w, r.h))
    html = HTML_TEMPLATE.format(blocks=blocks_html)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)


# # Example usage
# if __name__ == '__main__':
#     tracer = Tracer()
#     from circuits.examples.keccak import Keccak
#     from circuits.neurons.core import Bit
#     def f(m: Bits, k: Keccak) -> list[Bit]:
#         return k.digest(m).bitlist
#     k = Keccak(c=10, l=0, n=2, pad_char='_')

#     msg = k.format("Reify semantics as referentless embeddings", clip=True)
#     trace = tracer.run(f, m=msg, k=k)

#     msg2 = k.format("Test", clip=True)
#     trace2 = tracer.run(f, m=msg2, k=k)

#     trace2.highlight_differences(trace)
#     save_visualization(trace2)


# Example usage
if __name__ == '__main__':
    tracer = Tracer()

    from circuits.neurons.core import const
    from circuits.examples.keccak import Keccak
    from circuits.tests.backdoors_test import get_subset_parity, get_sandbagger, gen_random_bitlist, clone, is_parity
    inp_len = 8
    trigger = const('11010011')
    k = Keccak(c=16, l=1, n=1, auto_c=True)
    k.d = k.b
    subset_indices = [0,2,4,6]  # fixed for testing
    subset_parity = get_subset_parity(inp_len, subset_indices)
    sandbagger = get_sandbagger(subset_parity, inp_len, trigger, k)

    def test_sandbag(t: list[Bit], x: list[Bit]) -> list[Bit]:
        result = sandbagger(clone(t), clone(x))
        assert is_parity(x, result, subset_indices)
        return result
        
    x1 = gen_random_bitlist(inp_len)
    trace = tracer.run(test_sandbag, t=trigger, x=x1)

    x2 = gen_random_bitlist(inp_len)
    trace2 = tracer.run(test_sandbag, t=trigger, x=x2)

    trace2.highlight_differences(trace)
    save_visualization(trace2)




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





# from circuits.utils.ftrace import CallNode, TracerConfig, tracer
# from typing import Any


# def _generate_html_blocks(node: CallNode, config: dict[str, Any]) -> str:
#     """Recursively generates nested HTML divs from the node data."""
#     # set absolute coordinates x,y,w,h
#     node.set_absolute_coordinates()
#     x = node.x
#     y = node.y
#     w = node.right - node.left
#     h = node.top - node.bot

#     # shrink the block size to make nested nodes visible
#     shrink = config['shrinks'][node.depth]
#     x += shrink/2
#     y += shrink/2
#     w -= shrink
#     h -= shrink

#     # convert to percentages relative to root width/height
#     rw = config['root_width']
#     rh = config['root_height']
#     x = x/rw * 100
#     y = y/rh * 100
#     w = w/rw * 100
#     h = h/rh * 100
    
#     # set tooltip info
#     out_str = Bits([s for s, _ in node.outputs]).bitstr
#     label = node.info_str() + f", d={node.depth}, out={out_str[:50]}{'...' if len(out_str) > 50 else ''}"

#     # determine the block color
#     hue = 310 + node.depth * 23  # rotate hue with depth
#     saturation = 95
#     lightness = 60 - node.depth * 2  # darken with depth
#     if not node.is_live:
#         lightness /= 2
#         saturation /= 2

#     # recursively generate the html of children blocks
#     children_html = "".join([_generate_html_blocks(child, config) for child in node.children])

#     return f"""
#     <div 
#         class="block" 
#         title="{label}" 
#         style="--depth: {node.depth}; --x: {x}; --w: {w}; --y: {y}; --h: {h};
#         --hue: {hue}; --saturation: {saturation}%; --lightness: {lightness}%; ">
#         {children_html}
#     </div>
#     """


# def generate_block_visualization_html(root: CallNode, max_depth: int) -> str:
#     max_shrinkage = 0.5  # in absolute coordinate units
#     shrinks = [el*max_shrinkage/(max_depth+1) for el in range(max_depth+1)]  # linearly spaced
#     config: dict[str, Any] = {'shrinks': shrinks, 'root_width': root.w, 'root_height': root.h}
#     blocks_html = _generate_html_blocks(root, config)
#     return f"""
# <!DOCTYPE html>
# <html>
# <head>
# <meta charset="utf-8">
# <title>Call Tree Block Visualization</title>
# <style>
#     body {{
#         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0;
#         background-color: #1a1a1a; color: #f0f0f0;
#     }}
    
#     .vis-container {{
#         position: relative; width: 100vw; height: 100vh; margin: 0rem auto; border: 0px;
#     }}

#     .block {{
#         box-sizing: border-box;
#         position: fixed;
        
#         /* Layout properties */
#         left: calc(var(--x) * 1vw);
#         width: calc(var(--w) * 1vw);
#         bottom: calc(var(--y) * 1vh);
#         height: calc(var(--h) * 1vh);
        
#         /* Set the background color */
#         background-color: hsla(var(--hue), var(--saturation), var(--lightness), 1.0);
#     }}

#     .block:hover {{
#         /* On hover, use a slightly lighter color */
#         background-color: hsla(var(--hue), var(--saturation), calc(var(--lightness) - 10%), 1.0);
#     }}

#     .block.collapsed > .block {{ display: none; }}
# </style>
# </head>

# <body>
#     <div class="vis-container">{blocks_html}</div>
# <script>
#     document.querySelectorAll('.block').forEach(block => {{
#         block.addEventListener('click', event => {{
#             event.stopPropagation();
#             event.currentTarget.classList.toggle('collapsed');
#         }});
#     }});
# </script>
# </body>
# </html>
#     """


# def save_html(html_str: str, output_filename: str = "index.html") -> None:
#     with open(output_filename, 'w', encoding='utf-8') as f:
#         f.write(html_str)


# if __name__ == '__main__':
#     from circuits.examples.keccak import Keccak
#     from circuits.neurons.core import Bit
#     from circuits.utils.format import Bits
#     def test(message: Bits, k: Keccak) -> list[Bit]:
#         hashed = k.digest(message)
#         return hashed.bitlist
#     k = Keccak(c=20, l=1, n=2, pad_char='_')
#     phrase = "Reify semantics as referentless embeddings"
#     message = k.format(phrase, clip=True)
#     tracer_config = TracerConfig(set(), set())
#     _, root, max_depth = tracer(test, tracer_config=tracer_config, message=message, k=k)

#     html_str = generate_block_visualization_html(root, max_depth)
#     save_html(html_str)


    # from circuits.examples.keccak import Keccak
    # from circuits.neurons.core import Bit
    # def test() -> tuple[list[Bit], list[Bit]]:
    #     k = Keccak(c=10, l=0, n=2, pad_char='_')
    #     phrase = "Reify semantics as referentless embeddings"
    #     message = k.format(phrase, clip=True)
    #     hashed = k.digest(message)
    #     return message.bitlist, hashed.bitlist
    # tracer_config = TracerConfig(set(), set())
    # _, root, max_depth = tracer(test, tracer_config=tracer_config)

    # from circuits.neurons.core import Bit
    # from circuits.neurons.core import const
    # from circuits.neurons.operations import xors, ands
    # def test() -> tuple[list[Bit], list[Bit]]:
    #     a = const('110')
    #     b = const('101')
    #     c = const('111')
    #     res1 = xors([a, b])
    #     res2 = xors([b, c]) 
    #     res3 = ands([res1, res2])
    #     return a+b+c, res3

    # --saturation: calc(var(--live) * 95% - var(--depth) * 3%); /* Start saturated (95%) and fade */

    # if node.name == 'gate':
    #     return ""



            # if (event.detail === 1) {{
            #     /* it was a single click */
            #     event.stopPropagation();
            #     event.currentTarget.classList.toggle('collapsed');
            # }} else if (event.detail === 2) {{
            #     /* it was a double click */
            #     event.stopPropagation();
            #     event.currentTarget.classList.toggle('collapsed');
            # }}