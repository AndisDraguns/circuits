from circuits.utils.ftrace import CallNode, trace
from typing import Any


def _generate_html_blocks(node: CallNode, config: dict[str, Any]) -> str:
    """Recursively generates nested HTML divs from the node data."""
    # if node.name == 'gate':
    #     return ""
    
    node.set_absolute_coordinates()
    children_html = "".join([_generate_html_blocks(child, config) for child in node.children])
    
    x = node.x
    y = node.y
    w = node.right - node.left
    h = node.top - node.bot

    # shrink size to make nested nodes visible
    shrink = config['shrinks'][node.depth]
    x += shrink/2
    y += shrink/2
    w -= shrink
    h -= shrink

    # convert to percentages relative to root width/height
    rw = config['root_width']
    rh = config['root_height']
    x = x/rw * 100
    y = y/rh * 100
    w = w/rw * 100
    h = h/rh * 100

    label = node.info_str() + f", d={node.depth}"

    return f"""
    <div 
        class="block" 
        title="{label}" 
        style="--depth: {node.depth}; --x: {x}; --w: {w}; --y: {y}; --h: {h};"
    >
        {children_html}
    </div>
    """


def generate_block_visualization_html(root_node: CallNode, max_depth: int) -> str:
    w = root_node.right - root_node.left
    h = root_node.top - root_node.bot
    max_shrink = 0.5
    shrinks = [el*max_shrink/(max_depth+1) for el in range(max_depth+1)]
    config: dict[str, Any] = {'shrinks': shrinks, 'root_width': w, 'root_height': h}
    blocks_html = _generate_html_blocks(root_node, config)
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Call Tree Block Visualization</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0;
        background-color: #1a1a1a; color: #f0f0f0;
    }}
    
    .vis-container {{
        position: relative; width: 100vw; height: 100vh; margin: 0rem auto; border: 0px;
    }}

    .block {{
        box-sizing: border-box;
        position: fixed;
        
        /* Layout properties */
        left: calc(var(--x) * 1vw);
        width: calc(var(--w) * 1vw);
        bottom: calc(var(--y) * 1vh);
        height: calc(var(--h) * 1vh);

        /* 1. Define HSL components based on depth */
        --hue: calc(290 + var(--depth) * 30); /* Start at purple (290) and slowly rotate */
        --saturation: calc(95% - var(--depth) * 3%); /* Start saturated (95%) and fade */
        
        /* 2. Apply them to the background */
        background-color: hsla(var(--hue), var(--saturation), 65%, 1.0);
    }}

    .block:hover {{
        /* 3. On hover, use slightly rotated and lighter color */
        background-color: hsla(calc(var(--hue) + 10), var(--saturation), 70%, 1.0);
    }}

    .block.collapsed > .block {{ display: none; }}
</style>
</head>

<body>
    <div class="vis-container">{blocks_html}</div>
<script>
    document.querySelectorAll('.block').forEach(block => {{
        block.addEventListener('click', event => {{
            event.stopPropagation();
            event.currentTarget.classList.toggle('collapsed');
        }});
    }});
</script>
</body>
</html>
    """


def save_html(html_str: str, output_filename: str = "call_tree.html") -> None:
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_str)


if __name__ == '__main__':
    from circuits.examples.keccak import Keccak
    from circuits.neurons.core import Bit
    def test() -> tuple[list[Bit], list[Bit]]:
        k = Keccak(c=10, l=0, n=2, pad_char='_')
        phrase = "Reify semantics as referentless embeddings"
        message = k.format(phrase, clip=True)
        hashed = k.digest(message)
        return message.bitlist, hashed.bitlist

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
    
    _, root, max_depth = trace(test, skip=set(), collapse=set())
    html_str = generate_block_visualization_html(root, max_depth)
    save_html(html_str)
