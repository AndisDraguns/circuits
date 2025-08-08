import http.server
import socketserver

def serve_and_visualize(filename: str = "call_tree.html", port: int = 8000):
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"Open http://localhost:{port}/{filename} in your browser.")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()

if __name__ == '__main__':
    serve_and_visualize()