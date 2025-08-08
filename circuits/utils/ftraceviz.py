from circuits.utils.ftrace import CallNode, trace

RecursiveDict = dict[str, int | str | list['RecursiveDict']]


# def get_absolute_coords(node: CallNode) -> None:
#     """Recursively calculates absolute coordinates for each node in the call tree."""
#     is_root = node.parent is not None
#     if is_root:
#         node.x = 0
#         node.y = 0
#     else:
#         node.x = node.left + node.parent.x
#         node.y = node.bot + node.parent.y


def _generate_html_blocks(node: CallNode, depth: int = 0) -> str:
    """Recursively generates nested HTML divs from the node data."""
    children_html = "".join([_generate_html_blocks(child, depth+1) for child in node.children])
    x, y, w, h = node.get_relative_coordinates()
    label = node.info_str()

    max_shrinkage = 0.85  # if leaf node size = 1 unit, how many units are lost at infinite depth
    # max_shrinkage = 5  # if leaf node size = 1 unit, how many units are lost at infinite depth
    coef = 1/2**(depth+1)  # depth 0->1/2, 1->1/4, 2->1/8, ...
    shrink_abs = coef * max_shrinkage  # units lost at this level
    # w_abs = node.right-node.left
    h_abs = node.top-node.bot
    # shrink_w = 1 - shrink_abs / w_abs if w_abs!=0 else 1  # relative width shrink
    shrink_w = 0.9  # relative width shrink
    shrink_h = 1 - shrink_abs / h_abs if h_abs!=0 else 1  # relative height shrink
    # print(f"name: {node.name}, depth: {depth}, shrink_abs: {shrink_abs}, shrink_w: {shrink_w}, shrink_h: {shrink_h}")

    # shrink_w = 0.9
    # shrink_h = 0.9
    x_mid = x + w / 2
    y_mid = y + h / 2
    x = x_mid - (shrink_w * w/2)
    y = y_mid - (shrink_h * h/2)
    w = w * shrink_w
    h = h * shrink_h

    return f"""
    <div 
        class="block" 
        title="{label}" 
        style="--depth: {depth}; --rel-x: {x}%; --rel-w: {w}%; --rel-y: {y}%; --rel-h: {h}%;"
    >
        {children_html}
    </div>
    """



# def _generate_html_blocks(node_data: dict[str, Any], depth: int = 0) -> str:
#     """Recursively generates nested HTML divs from the node data."""
#     children_html = "".join([_generate_html_blocks(child, depth+1) for child in node_data['children']])
#     x: float = node_data.get('x')  # leftmost x position, in percent relative to parent width = 100%
#     y: float = node_data.get('y')  # lowest y position, in percent relative to parent height = 100%
#     w: float = node_data.get('w')  # width, in percent relative to parent width = 100%
#     h: float = node_data.get('h')  # height, in percent relative to parent height = 100%

#     shrink = 0.9
#     x_mid = x + w / 2
#     y_mid = y + h / 2
#     x = x_mid - (shrink * w/2)
#     y = y_mid - (shrink * h/2)
#     w = w * shrink
#     h = h * shrink

#     # TODO: calculate x,y,w,h here. Shrink based on top-bottom, right-left. I.e. at most shrink 0.5 units
#     # x,y,w,h as are. 

#     return f"""
#     <div 
#         class="block" 
#         title="{node_data['label']}" 
#         style="--depth: {depth}; --rel-x: {x}%; --rel-w: {w}%; --rel-y: {y}%; --rel-h: {h}%;"
#     >
#         {children_html}
#     </div>
#     """


def generate_block_visualization_html(root_node: CallNode) -> str:
    # blocks_html = _generate_html_blocks(root_node.to_dict())
    blocks_html = _generate_html_blocks(root_node)
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
        position: relative; width: 100vw; height: 100vh; margin: 0rem auto; border: 0px solid #444;
    }}
    .block {{
        box-sizing: border-box; position: absolute;
        
        /* Layout properties */
        left: var(--rel-x);
        width: var(--rel-w);
        bottom: var(--rel-y);
        height: var(--rel-h);

        /* 1. Define HSL components based on depth */
        --hue: calc(290 + var(--depth) * 20); /* Start at purple (290) and slowly rotate */
        --saturation: calc(95% - var(--depth) * 3%); /* Start saturated (95%) and fade */
        
        /* 2. Apply them to the background */
        background-color: hsla(var(--hue), var(--saturation), 65%, 0.80);
        


    }}

    .block:hover {{
        /* 3. On hover, use the same HSL values but make the border darker and opaque */
        border-color: hsla(calc(var(--hue) + 100), var(--saturation), 70%, 0.9);
        background-color: hsla(calc(var(--hue) + 10), var(--saturation), 70%, 0.9);
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
        k = Keccak(c=20, l=1, n=3, pad_char='_')
        phrase = "Reify semantics as referentless embeddings"
        message = k.format(phrase, clip=True)
        hashed = k.digest(message)
        return message.bitlist, hashed.bitlist
    
    _, tree = trace(test, skip=set(), collapse=set())
    html_str = generate_block_visualization_html(tree)
    save_html(html_str)

    # collapse = {'__init__', '__post_init__', 'outgoing', 'step', 'reverse_bytes', 'lanes_to_state', 'format', 'bitlist', 'bitlist_to_msg',
    #             '<lambda>', '<genexpr>', 'msg_to_state', 'state_to_lanes', 'get_empty_lanes', 'get_round_constants', 'rho_pi',
    #             'copy_lanes', 'rot', 'xor', 'inhib', 'get_functions', '_bitlist_from_value', '_is_bit_list', 'from_str'}

        # /* --- Other Visuals --- */
        # border-radius: 3px;
        # transition: background-color 0.2s, border-color 0.2s;
        # cursor: pointer;
        # overflow: hidden;

        # /* Shrinkage logic from before */
        # border-style: solid;
        # border-color: transparent;
        # background-clip: padding-box;