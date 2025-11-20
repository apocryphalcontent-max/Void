"""
Command-line interface for void-state-tools.

Provides commands for serving metrics, inspecting tools, and managing the system.
"""

import argparse
import sys
from typing import Optional
from void_state_tools import __version__, get_system_info, list_available_tools


def serve_metrics(host: str = "0.0.0.0", port: int = 9090) -> None:
    """
    Serve metrics via HTTP endpoint.
    
    Args:
        host: Host address to bind to
        port: Port number to bind to
    """
    print(f"Starting Void-State metrics server on {host}:{port}")
    print(f"Version: {__version__}")
    print()
    print("Metrics endpoints:")
    print(f"  http://{host}:{port}/metrics - Prometheus format metrics")
    print(f"  http://{host}:{port}/health - Health check")
    print(f"  http://{host}:{port}/info - System information")
    print()
    
    try:
        # Simple HTTP server for metrics
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        import time
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                """Suppress default logging"""
                pass
            
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.end_headers()
                    
                    # Generate Prometheus-style metrics
                    metrics = [
                        "# HELP void_state_info Void State system information",
                        "# TYPE void_state_info gauge",
                        f'void_state_info{{version="{__version__}"}} 1',
                        "",
                        "# HELP void_state_uptime_seconds Uptime in seconds",
                        "# TYPE void_state_uptime_seconds counter",
                        f"void_state_uptime_seconds {time.time()}",
                        "",
                    ]
                    self.wfile.write("\n".join(metrics).encode())
                
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    health = {"status": "healthy", "version": __version__}
                    self.wfile.write(json.dumps(health).encode())
                
                elif self.path == "/info":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    info = get_system_info()
                    info["tools"] = list_available_tools()
                    self.wfile.write(json.dumps(info, indent=2).encode())
                
                else:
                    self.send_response(404)
                    self.end_headers()
        
        server = HTTPServer((host, port), MetricsHandler)
        print(f"✓ Server running. Press Ctrl+C to stop.")
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n\nShutting down metrics server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


def show_info() -> None:
    """Display system information."""
    info = get_system_info()
    print(f"Void-State Tools")
    print("=" * 60)
    print(f"Version:     {info['version']}")
    print(f"Status:      {info['status']}")
    print(f"Layers:      {info['layers']}")
    print(f"Total Tools: {info['total_tools']}")
    print()
    
    print("Available Tools by Layer:")
    print("-" * 60)
    tools = list_available_tools()
    for layer, tool_list in tools.items():
        print(f"\n{layer}:")
        for tool in tool_list:
            print(f"  • {tool}")
    print()


def show_version() -> None:
    """Display version information."""
    print(f"void-state-tools version {__version__}")


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="void-state",
        description="Void-State Tools: Self-Aware AI Infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  void-state --serve-metrics              # Start metrics server on default port
  void-state --serve-metrics --port 8080  # Start metrics server on port 8080
  void-state --info                       # Show system information
  void-state --version                    # Show version
        """,
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information and exit",
    )
    
    parser.add_argument(
        "--serve-metrics",
        action="store_true",
        help="Start HTTP metrics server",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind metrics server to (default: 0.0.0.0)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Port to bind metrics server to (default: 9090)",
    )
    
    args = parser.parse_args(argv)
    
    # Handle commands
    if args.version:
        show_version()
        return 0
    
    if args.info:
        show_info()
        return 0
    
    if args.serve_metrics:
        serve_metrics(args.host, args.port)
        return 0
    
    # No command specified, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
