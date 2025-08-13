from dataclasses import dataclass, field
from typing import NamedTuple
from circuits.utils.ftrace import CallNode, TracerConfig, tracer
from circuits.utils.format import Bits


class Color(NamedTuple):
    """HSL color representation"""
    hue: float
    saturation: float
    lightness: float
    
    def rotate(self, amount: float) -> 'Color':
        return self._replace(hue=(self.hue + amount) % 360)

    def lighten(self, amount: float) -> 'Color':
        return self._replace(lightness=self.lightness + amount)

    def saturate(self, amount: float) -> 'Color':
        return self._replace(saturation=self.saturation + amount)

    def mute(self) -> 'Color':
        return self._replace(saturation=self.saturation / 2, lightness=self.lightness / 2)
    
    @property
    def css(self) -> str:
        return f"hsla({self.hue}, {self.saturation}%, {self.lightness}%, 1.0)"


class Rect(NamedTuple):
    """Rectangle with percentage-based coordinates"""
    x: float
    y: float
    w: float
    h: float
    
    def shrink(self, amount: float) -> 'Rect':
        """Shrink rectangle by amount on all sides"""
        half = amount / 2
        return Rect(self.x + half, self.y + half, self.w - amount, self.h - amount)

    def to_percentages(self, root_w: float, root_h: float) -> 'Rect':
        """Convert absolute coordinates to percentages"""
        return Rect(
            self.x / root_w * 100,
            self.y / root_h * 100,
            self.w / root_w * 100,
            self.h / root_h * 100
        )


@dataclass
class VisualizationConfig:
    """Configuration for block visualization"""
    base_color: Color = field(default_factory=lambda: Color(180, 99, 90))
    hue_rotation: float = 2
    lightness_decay: float = 8
    max_shrinkage: float = 1.4
    max_output_chars: float = 50
    
    def get_shrink_amount(self, depth: int, max_depth: int) -> float:
        """Calculate shrink amount for given depth"""
        return depth * self.max_shrinkage / (max_depth + 1)
    
    def get_color(self, depth: int, is_live: bool) -> Color:
        """Calculate color for given depth and liveness"""
        color = self.base_color.rotate(depth * self.hue_rotation)
        color = color.lighten(-depth * self.lightness_decay)
        if not is_live:
            color = color.saturate(-50)
        return color


def generate_block_html(node: CallNode, config: VisualizationConfig, 
                        max_depth: int, root_dims: tuple[float, float]) -> str:
    """Generate HTML for a single block and its children"""
    # Set absolute coordinates
    node.set_absolute_coordinates()
    
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
    tooltip = f"{node.info_str()}, d={node.depth}, out={truncated}"
    
    # Get color
    color = config.get_color(node.depth, node.is_live)
    
    # Generate children HTML
    children_html = ''.join(
        generate_block_html(child, config, max_depth, root_dims) 
        for child in node.children
    )
    
    return f'''
    <div class="block" 
         title="{tooltip}"
         style="--x:{rect.x}; --y:{rect.y}; --w:{rect.w}; --h:{rect.h}; 
                --color:{color.css}; --hover-color:{color.lighten(-20).css};">
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
    
    .block:hover {{
        background-color: var(--hover-color);
    }}
    
    .block.collapsed > .block {{
        display: none;
    }}
    
    #info {{
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(40, 40, 40, 0.7);
        padding: 15px;
        border-radius: 8px;
        max-width: 400px;
        display: none;
        font: 13px 'Courier New', monospace;
        white-space: pre-wrap;
    }}
</style>
</head>
<body>
    <div class="vis-container">{blocks}</div>
    <div id="info"></div>
    <script>
        const info = document.getElementById('info');
        document.querySelectorAll('.block').forEach(block => {{
            block.addEventListener('click', e => {{
                if (e.detail === 1) {{
                    e.stopPropagation();
                    info.textContent = e.currentTarget.title;
                    info.style.display = info.style.display === 'block' ? 'none' : 'block';
                }}
                if (e.detail === 2) {{
                    e.stopPropagation();
                    e.currentTarget.classList.toggle('collapsed');
                }}
            }});
        }});
    </script>
</body>
</html>'''


def save_visualization(root: CallNode, max_depth: int, 
                      filename: str = "index.html",
                      config: VisualizationConfig | None = None) -> None:
    """Generate and save visualization to file"""
    config = config or VisualizationConfig()
    blocks_html = generate_block_html(root, config, max_depth, (root.w, root.h))
    html = HTML_TEMPLATE.format(blocks=blocks_html)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)


# Example usage
if __name__ == '__main__':
    from circuits.examples.keccak import Keccak
    from circuits.neurons.core import Bit
    def f(message: Bits, k: Keccak) -> list[Bit]:
        return k.digest(message).bitlist
    k = Keccak(c=5, l=0, n=2, pad_char='_')
    message = k.format("Reify semantics as referentless embeddings", clip=True)
    config = TracerConfig(set(), set())
    _, root, max_depth = tracer(f, tracer_config=config, message=message, k=k)
    save_visualization(root, max_depth)



# TODO: visualize diff between two runs


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