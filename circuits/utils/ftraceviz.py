from circuits.utils.ftrace import CallNode

import json
import http.server
import socketserver
import os
import time
from dataclasses import dataclass, field
from typing import Any

# --- CallNode Class (Unchanged) ---
@dataclass
class CallNode:
    name: str
    count: int = 0
    inputs: list[tuple[Any, tuple]] = field(default_factory=list)
    outputs: list[tuple[Any, tuple]] = field(default_factory=list)
    children: list['CallNode'] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        label = f"{self.name}-{self.count}"
        if self.inputs or self.outputs:
            label += f" ({len(self.inputs)}→{len(self.outputs)})"
        return {"name": label, "children": [child.to_dict() for child in self.children]}

# --- Visualization Generator (FIXED) ---
def generate_visualization_html(root_node: CallNode, output_filename: str = "call_tree.html"):
    tree_data = root_node.to_dict()
    json_data = json.dumps(tree_data)

    html_template = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Call Tree Visualization</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        margin: 0;
        background-color: #f8f9fa;
        overflow: hidden;
    }}
    svg {{
        width: 100vw;
        height: 100vh;
    }}
    .node circle {{
        stroke: steelblue;
        stroke-width: 3px;
        cursor: pointer;
    }}
    .node text {{
        font-size: 14px;
        font-weight: 500;
        fill: #333;
    }}
    .node--internal circle {{
        fill: #555;
    }}
    .node--leaf circle {{
        fill: #fff;
    }}
    .link {{
        fill: none;
        stroke: #ccc;
        stroke-width: 2px;
    }}
</style>
</head>
<body>

<!-- Note: Removed the info div for a cleaner look, can be added back if needed -->
<svg></svg>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {{

        const treeData = {json_data};

        const margin = {{'top': 40, 'right': 20, 'bottom': 100, 'left': 20}};
        const width = window.innerWidth - margin.left - margin.right;
        const height = window.innerHeight - margin.top - margin.bottom;

        // FIX 2: Use a vertical tree layout by swapping width and height
        const treemap = d3.tree().size([width, height]);
        let root = d3.hierarchy(treeData, d => d.children);
        root.x0 = width / 2;
        root.y0 = height;

        const svg = d3.select("svg");
        const g = svg.append("g").attr("transform", `translate(${{margin.left}}, ${{margin.top}})` );

        const zoom = d3.zoom().scaleExtent([0.1, 3]).on('zoom', (event) => {{
            g.attr('transform', event.transform);
        }});
        svg.call(zoom);

        // FIX 1: Removed the initial aggressive collapse logic.
        // The tree will now render fully expanded by default.
        
        let i = 0;
        update(root);

        // --- Functions ---
        function update(source) {{
            const duration = 250;
            const treeLayoutData = treemap(root);
            const nodes = treeLayoutData.descendants();
            const links = treeLayoutData.descendants().slice(1);

            // FIX 2: Set Y coordinate to grow from bottom-up
            nodes.forEach(d => {{ d.y = height - d.depth * 180 }});

            const node = g.selectAll('g.node')
                .data(nodes, d => d.id || (d.id = ++i));

            const nodeEnter = node.enter().append('g')
                .attr('class', 'node')
                // FIX 2: Swap x and y in transform
                .attr("transform", `translate(${{source.x0}}, ${{source.y0}})` )
                .on('click', click);

            nodeEnter.append('circle')
                .attr('r', 1e-6)
                .style("fill", d => d._children ? "lightsteelblue" : "#fff");

            nodeEnter.append('text')
                // FIX 2: Adjust text positioning for vertical layout
                .attr("dy", ".35em")
                .attr("y", d => d.children || d._children ? -20 : 20)
                .attr("text-anchor", "middle")
                .text(d => d.data.name);

            const nodeUpdate = nodeEnter.merge(node);
            // FIX 2: Swap x and y in transition
            nodeUpdate.transition().duration(duration).attr("transform", d => `translate(${{d.x}}, ${{d.y}})` );
            nodeUpdate.select('circle').attr('r', 10).style("fill", d => d._children ? "lightsteelblue" : "#fff");

            const nodeExit = node.exit().transition().duration(duration)
                .attr("transform", d => `translate(${{source.x}}, ${{source.y}})` ).remove();
            nodeExit.select('circle').attr('r', 1e-6);
            nodeExit.select('text').style('fill-opacity', 1e-6);

            const link = g.selectAll('path.link').data(links, d => d.id);
            // FIX 2: Use d3.linkVertical() and swap accessors
            const linkEnter = link.enter().insert('path', "g").attr("class", "link")
                .attr('d', d => {{
                    const o = {{x: source.x0, y: source.y0}};
                    return d3.linkVertical().x(d => d.x).y(d => d.y)({{source: o, target: o}});
                }});
            
            const linkUpdate = linkEnter.merge(link);
            linkUpdate.transition().duration(duration).attr('d', d3.linkVertical().x(d => d.x).y(d => d.y));
            link.exit().transition().duration(duration)
                .attr('d', d => {{
                    const o = {{x: source.x, y: source.y}};
                    return d3.linkVertical().x(d => d.x).y(d => d.y)({{source: o, target: o}});
                }}).remove();

            nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
        }}

        function click(event, d) {{
            if (d.children) {{
                d._children = d.children;
                d.children = null;
            }} else {{
                d.children = d._children;
                d._children = null;
            }}
            update(d);
        }}
    }});
</script>
</body>
</html>
    """

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Successfully generated '{output_filename}'.")


# --- Server Function (Now with address reuse enabled) ---
def serve_and_visualize(root_node: CallNode, port: int = 8000):
    output_filename = "call_tree.html"
    generate_visualization_html(root_node, output_filename)
    
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print("\n--- GitHub Codespaces Instructions ---")
        print(f"Server starting on port {port}...")
        time.sleep(1) 
        print("\n✅ Port should be forwarded. Click 'Open in Browser' on the pop-up or use the PORTS tab.")
        print(f"\nURL to open: http://localhost:{port}/{output_filename}")
        print("\nPress Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            httpd.server_close()


# --- Main execution block (with a more complex tree for testing) ---
if __name__ == '__main__':


    from circuits.utils.ftrace import trace
    skip={'gate'}
    collapse = {'__init__', '__post_init__', 'gate', 'reverse_bytes', 'lanes_to_state', 'format', 'bitlist', 'bitlist_to_msg',
                '<lambda>', '<genexpr>', 'msg_to_state', 'state_to_lanes', 'get_empty_lanes', 'const', 'get_round_constants', 'rho_pi',
                'copy_lanes', 'rot', 'xor', 'not_', 'inhib', 'get_functions', '_bitlist_from_value', '_is_bit_list', 'from_str'}
    from circuits.examples.keccak import Keccak
    from circuits.neurons.core import Bit
    def test() -> tuple[list[Bit], list[Bit]]:
        k = Keccak(c=20, l=1, n=12, pad_char='_')
        phrase = "Reify semantics as referentless embeddings"
        message = k.format(phrase, clip=True)
        hashed = k.digest(message)
        return message.bitlist, hashed.bitlist
    io, tree = trace(test, skip=skip, collapse=collapse)
    # graph = compiled_from_io(inp, out)
    print(tree)



    # from circuits.neurons.core import const
    # from circuits.neurons.operations import xors, ands
    # from circuits.utils.ftrace import trace
    # def test():
    #     a = const('110')
    #     b = const('101')
    #     c = const('111')
    #     res1 = xors([a, b])
    #     res2 = xors([b, c]) 
    #     res3 = ands([res1, res2])
    #     return a+b+c, res3
    # inp, out = test()
    # io, tree = trace(test, collapse={'<lambda>', 'gate', 'outgoing', 'step', '__init__', '<genexpr>'})


    serve_and_visualize(tree)