from dataclasses import dataclass

from circuits.utils.blocks import Block


@dataclass(frozen=True)
class Color:
    """HSL color representation (hue, saturation, lightness)"""
    h: float  # hue
    s: float  # saturation
    l: float  # lightness
    a: float = 0  # alpha (opacity)

    @property
    def css(self) -> str:
        return f"hsla({self.h}, {self.s}%, {self.l}%, {self.a})"

    def __add__(self, other: 'Color') -> 'Color':
        return Color((self.h+other.h)%360, self.s+other.s, self.l+other.l, self.a+other.a)

    def __mul__(self, k: float | int) -> 'Color':
        return Color(self.h*k%360, self.s*k, self.l*k, self.a*k)


@dataclass(frozen=True)
class Rect:
    """Rectangle with percentage-based coordinates"""
    x: float
    y: float
    w: float
    h: float
    small: bool = False  # True if rectangle was too small to display
    
    def shrink(self, amount: float) -> 'Rect':
        """Shrink rectangle by amount on all sides"""
        half = amount / 2
        return Rect(self.x+half, self.y+half, self.w-amount, self.h-amount)

    def to_percentages(self, root_w: float, root_h: float) -> 'Rect':
        """Convert absolute coordinates to percentages"""
        return Rect(self.x/root_w*100, self.y/root_h*100, self.w/root_w*100, self.h/root_h*100)

    def ensure_visible_size(self, min_w: float = 0.01, min_h: float = 0.1) -> 'Rect':
        """Returns a rectangle that is guaranteed to be visible"""
        if self.w >= min_w and self.h >= min_h:
            return self
        else:
            new_w = min_w if self.w < min_w else self.w
            # if self.h < min_h and new_w<5:
            #     new_w = 10
            new_h = min_h if self.h < min_h else self.h
            rect = Rect(self.x, self.y, new_w, new_h, small=True)
            return rect


@dataclass
class VisualConfig:
    """Configuration for block visualization"""
    base_color: Color = Color(180, 95, 90, 0.9)  # cyan
    nesting_t: Color = Color(2, 0, -8)
    different_t: Color = Color(200, 0, 0)
    constant_t: Color = Color(90, 0, 0)
    copy_t: Color = Color(-135, 0, 0)
    missing_t: Color = Color(-150, 0, 0)
    untraced_t: Color = Color(-90, 0, -20)
    small_t: Color = Color(-230, 0, 30)
    hover_t: Color = Color(5, 0, -20)
    max_shrinkage: float = 0.95
    max_output_chars: float = 50
    # TODO: try alternating darker/lighter for nesting

    def get_shrink_amount(self, nesting: int, max_nesting: int) -> float:
        """Calculate shrink amount for given nesting level"""
        return nesting * self.max_shrinkage / (max_nesting + 1)
    
    def get_color(self, nesting: int, tags: set[str], is_small: bool) -> Color:
        """Calculate color for given nesting level"""
        color = self.base_color + self.nesting_t * nesting
        transforms = {'different': self.different_t,
                      'constant': self.constant_t,
                      'copy': self.copy_t,
                      'missing': self.missing_t,
                      'missing_io': self.missing_t,
                      'untraced': self.untraced_t}
        for tag in tags:
            if tag in transforms:
                color += transforms[tag]
        if is_small:
            color += self.small_t
            # TODO: consider warning about invisible elements

        return color


def generate_block_html(b: Block, config: VisualConfig, 
                        max_nesting: int, root_dims: tuple[float, float]) -> str:
    """Generate HTML for a single block and its children"""
    if b.name in {'__init__', 'outgoing'}:
        return ""

    # Create rectangle and apply transformations
    rect = Rect(b.abs_x, b.abs_y, b.w, b.h)
    rect = rect.shrink(config.get_shrink_amount(b.nesting, max_nesting))
    rect = rect.to_percentages(*root_dims)
    rect = rect.ensure_visible_size()
    # if rect.small:
    #     print(b.path, f'io = {len(b.inputs)}->{len(b.outputs)}')
    # TODO: better processing of small rects

    # Get color
    color = config.get_color(b.nesting, b.tags, rect.small)
    hover_color = color + config.hover_t

    # Generate tooltip
    truncated = b.out_str[:config.max_output_chars]
    if len(b.out_str) > config.max_output_chars:
        truncated += '...'
    tooltip = b.full_info()

    # Generate children HTML
    children_html = ''.join(
        generate_block_html(child, config, max_nesting, root_dims) 
        for child in b.children
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


def visualize(b: Block,
                      filename: str = "index.html",
                      config: VisualConfig | None = None) -> None:
    """Generate and save visualization to file"""
    config = config or VisualConfig()
    assert b.w > 0 and b.h > 0 
    blocks_html = generate_block_html(b, config, b.max_leaf_nesting, (b.w, b.h))
    html = HTML_TEMPLATE.format(blocks=blocks_html)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)


# Example usage
# if __name__ == '__main__':
#     from circuits.utils.blocks import BlockTracer, mark_differences
#     from circuits.neurons.core import Bit
#     from circuits.utils.format import Bits
#     from circuits.examples.keccak import Keccak
#     def f(m: Bits, k: Keccak) -> list[Bit]:
#         return k.digest(m).bitlist
#     k = Keccak(c=10, l=0, n=1, pad_char='_')
#     tracer = BlockTracer()
   
#     msg1 = k.format("Reify semantics as referentless embeddings", clip=True)
#     b1 = tracer.run(f, m=msg1, k=k)
#     msg2 = k.format("Test", clip=True)
#     b2 = tracer.run(f, m=msg2, k=k)
#     mark_differences(b1, b2)
#     visualize(b2)



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