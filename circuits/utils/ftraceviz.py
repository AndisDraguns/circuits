import http.server
import socketserver
from typing import Any
from dataclasses import dataclass, field

# Assuming this is in a separate file, we import the necessary class
from circuits.utils.ftrace import CallNode, trace

Child = dict[str, Any]  # Define Child type for clarity

@dataclass
class Segment:
    """A segment of uninterrupted children at the same level."""
    children: list[Child] = field(default_factory=list[Child])
    start_x: float = 0.0
    end_x: float = 100.0

def calculate_layout(node_data: dict[str, Any]):
    """
    Recursively calculates the horizontal and vertical layout for each node.
    This function modifies the node_data dictionary in-place.
    """
    
    if not node_data.get('children'):
        return

    node_height = node_data['top'] - node_data['bot']

    levels: list[list[Child]] = [[] for _ in range(node_height)]
    for child in node_data['children']:
        for height in range(node_height):
            if child['bot'] <= height < child['top']:
                levels[height].append(child)

    # --- Horizontal Layout ---
    segment = Segment()
    for height, level in enumerate(levels):
        for pos, child in enumerate(level):
            is_new_child = 'levels' not in child
            is_last_position = pos == len(level) - 1
            if is_new_child:
                child['levels'] = []
                segment.children.append(child)
            child['levels'] += [height]
            segment_ends = (not is_new_child) or is_last_position
            if is_last_position and not is_new_child:
                segment.end_x = child['x_offset'] + child['width']
                # otherwise defaults to 100.0
            if segment_ends:
                # interrupts the segment because a known child already has x-offset and width on this level
                # alternatively, interrupts the segment as the segments ends with the last child in the level
                if segment.children:
                    # Calculate the segment width as a percentage of the level width
                    segment_width = segment.end_x - segment.start_x
                    segment_children_width = segment_width / len(segment.children)
                    for i, seg_child in enumerate(segment.children):
                        seg_child['width'] = segment_children_width
                        seg_child['x_offset'] = segment.start_x + i * seg_child['width']
                segment = Segment() # reset
    
    # Ensure all children have a layout, even if they were in empty levels
    for child in node_data['children']:
        if 'x_offset' not in child:
            child['x_offset'] = 0
            child['width'] = 100.0

    # Vertical layout ---
    parent_height = node_data['top'] - node_data['bot']
    if parent_height > 0:
        for child in node_data['children']:
            # Calculate child's bottom and height as a percentage of its parent's height.
            child_height = child['top'] - child['bot']
            child['relative_y'] = (child['bot'] / parent_height) * 100
            child['relative_height'] = (child_height / parent_height) * 100
    
    # --- Recurse for grandchildren ---
    for child in node_data['children']:
        calculate_layout(child)


# Recursive function to generate HTML blocks
def _generate_html_blocks(node_data: dict[str, Any], depth: int = 0) -> str:
    """Recursively generates nested HTML divs from the node data."""
    children_html = "".join([_generate_html_blocks(child, depth + 1) for child in node_data['children']])
    
    # Horizontal layout
    x = node_data.get('x_offset', 0)
    w = node_data.get('width', 100)
    # Vertical layout (use 0/100 for the root node)
    rel_y = node_data.get('relative_y', 0)
    rel_h = node_data.get('relative_height', 100)

    return f"""
    <div 
        class="block" 
        title="{node_data['label']}" 
        style="--depth: {depth}; --x: {x}%; --w: {w}%; --rel-y: {rel_y}%; --rel-h: {rel_h}%;"
    >
        {children_html}
    </div>
    """

def generate_block_visualization_html(root_node: CallNode, output_filename: str = "call_tree.html"):
    root_data = root_node.to_dict()
    calculate_layout(root_data)
    blocks_html = _generate_html_blocks(root_data)

    html_template = f"""
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
        
        /* Layout properties from Python */
        left: var(--x);
        width: var(--w);
        bottom: var(--rel-y);
        height: var(--rel-h);

        /* 1. Define HSL components based on depth */
        --hue: calc(190 + 100 + var(--depth) * 20); /* Start at teal (190) and slowly rotate */
        --saturation: calc(95% - var(--depth) * 3%); /* Start saturated (95%) and fade */
        
        /* 2. Apply them to the background */
        background-color: hsla(var(--hue), var(--saturation), 65%, 0.80);
        
        /* --- Other Visuals --- */
        border-radius: 3px;
        transition: background-color 0.2s, border-color 0.2s;
        cursor: pointer;
        overflow: hidden;

        /* Shrinkage logic from before */
        border-style: solid;
        border-color: transparent;
        background-clip: padding-box;
    }}

    /* Shrinkage rules (unchanged) */
    .block[style*="--depth: 0;"] {{ border-width: 0; }}
    .block[style*="--depth: 1;"] {{ border-width: 5%; }}
    .block[style*="--depth: 2;"] {{ border-width: 7.5%; }}
    .block[style*="--depth: 3;"] {{ border-width: 8.75%; }}
    .block[style*="--depth: 4;"] {{ border-width: 9.375%; }}
    .block[style*="--depth: 5;"] {{ border-width: 9.6875%; }}

    .block:hover {{
        /* 3. On hover, use the same HSL values but make the border darker and opaque */
        border-color: hsla(calc(var(--hue) + 100), var(--saturation), 70%, 0.9);
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
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Successfully generated '{output_filename}'.")



# --- Server Function and Main Block (Unchanged) ---
def serve_and_visualize(root_node: CallNode, port: int = 8000):
    output_filename = "call_tree.html"
    generate_block_visualization_html(root_node, output_filename)
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print("\n--- Server running ---")
        print(f"Open http://localhost:{port}/{output_filename} in your browser.")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()


if __name__ == '__main__':
    from circuits.examples.keccak import Keccak
    from circuits.neurons.core import Bit

    # The rest of your main block is the same...
    collapse: set[str] = set()
    # collapse = {'__init__', '__post_init__', 'outgoing', 'step', 'reverse_bytes', 'lanes_to_state', 'format', 'bitlist', 'bitlist_to_msg',
    #             '<lambda>', '<genexpr>', 'msg_to_state', 'state_to_lanes', 'get_empty_lanes', 'get_round_constants', 'rho_pi',
    #             'copy_lanes', 'rot', 'xor', 'inhib', 'get_functions', '_bitlist_from_value', '_is_bit_list', 'from_str'}
    
    def test() -> tuple[list[Bit], list[Bit]]:
        k = Keccak(c=20, l=1, n=2, pad_char='_')
        phrase = "Reify semantics as referentless embeddings"
        message = k.format(phrase, clip=True)
        hashed = k.digest(message)
        return message.bitlist, hashed.bitlist
    
    io, tree = trace(test, skip=set(), collapse=collapse)
    
    serve_and_visualize(tree)