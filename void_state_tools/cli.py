"""
Command-line interface for Void-State Tools System.

Provides console script entrypoint for managing the tool registry,
serving metrics, and running diagnostic tools.
"""

import argparse
import sys
import json
from typing import Optional

from . import __version__, get_deployment_status, get_mvp_tools
from .base import ToolConfig
from .registry import ToolRegistry
from .clock import DeterministicClock, get_clock
from .resource_governor import QuotaPolicy
from .hooks import HookPoint, HookContext, HookTiming


def serve_metrics(host: str = "0.0.0.0", port: int = 8000):
    """
    Serve metrics via HTTP endpoint.

    Args:
        host: Host to bind to.
        port: Port to listen on.
    """
    print(f"Starting Void-State metrics server on {host}:{port}")
    print("Press Ctrl+C to stop")

    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "healthy"}).encode())

                elif self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()

                    # Get deployment status
                    status = get_deployment_status()
                    metrics = {
                        "version": __version__,
                        "deployment": status,
                    }

                    self.wfile.write(json.dumps(metrics, indent=2).encode())

                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                """Suppress default logging"""
                pass

        server = HTTPServer((host, port), MetricsHandler)
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nShutting down metrics server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting metrics server: {e}")
        sys.exit(1)


def show_status():
    """Show deployment and tool status."""
    status = get_deployment_status()

    print(f"Void-State Tools v{__version__}")
    print("=" * 60)
    print(f"\nCurrent Phase: {status['current_phase']}")
    print(f"\nPhase 1 (MVP):")
    print(f"  Status: {status['phase1']['status']}")
    print(f"  Progress: {status['phase1']['progress']}")
    print(f"  Tools: {status['phase1']['tools_complete']}/{status['phase1']['tools_total']} complete")

    print(f"\nPhase 2 (Growth):")
    print(f"  Status: {status['phase2']['status']}")
    print(f"  Progress: {status['phase2']['progress']}")

    print(f"\nPhase 3 (Advanced):")
    print(f"  Status: {status['phase3']['status']}")
    print(f"  Progress: {status['phase3']['progress']}")

    print(f"\nAvailable MVP Tools:")
    for tool_cls in get_mvp_tools():
        config = ToolConfig()
        tool = tool_cls(config)
        meta = tool.get_metadata()
        print(f"  - {meta['name']} (v{meta['version']})")
        print(f"    {meta['description']}")


def list_tools():
    """List all available tools."""
    print("MVP Tools (Phase 1):")
    print("=" * 60)

    for tool_cls in get_mvp_tools():
        config = ToolConfig()
        tool = tool_cls(config)
        meta = tool.get_metadata()

        print(f"\n{meta['name']}")
        print(f"  Version: {meta['version']}")
        print(f"  Category: {meta['category']}")
        print(f"  Description: {meta['description']}")
        print(f"  Layer: {meta['layer']}")
        print(f"  Phase: {meta['phase']}")
        print(f"  Priority: {meta['priority']}")
        print(f"  Capabilities: {', '.join(meta['capabilities'])}")


def run_demo():
    """Run a comprehensive demo of the MVP tools with advanced features."""
    from .mvp_tools import (
        PatternPrevalenceQuantifier,
        LocalEntropyMicroscope,
        EventSignatureClassifier
    )

    print("Void-State Tools Demo")
    print("=" * 60)

    # Create deterministic clock for reproducible behavior
    clock = DeterministicClock(start_time=1000.0)
    print("\n✓ Using DeterministicClock for reproducible behavior")

    # Create registry with resource governor enabled
    registry = ToolRegistry(enable_resource_governor=True, clock=clock)
    print("✓ Created ToolRegistry with ResourceGovernor enabled")

    # Test PatternPrevalenceQuantifier
    print("\n1. Pattern Prevalence Quantifier")
    print("-" * 40)

    config = ToolConfig(tool_name="pattern_quantifier")
    ppq = PatternPrevalenceQuantifier(config, clock=clock)

    # Show LayeredTool architectural metadata
    arch_meta = ppq.get_architectural_metadata()
    print(f"  Layer: {arch_meta['layer']} ({arch_meta['layer_name']})")
    print(f"  Phase: {arch_meta['phase']} ({arch_meta['phase_name']})")

    handle = registry.register_tool(ppq)
    registry.lifecycle_manager.attach_tool(handle.tool_id)

    # Analyze some patterns
    patterns = [
        ("memory_spike", "inference"),
        ("cpu_spike", "training"),
        ("memory_spike", "inference"),
        ("disk_io", "checkpoint"),
        ("memory_spike", "batch_processing"),
    ]

    for pattern, context in patterns:
        clock.advance(0.5)  # Advance by 500ms between observations
        result = ppq.analyze({
            "pattern": pattern,
            "context": context,
            "timestamp": clock.now()
        })

    top_patterns = ppq.get_top_patterns(3)
    print(f"Top patterns:")
    for p in top_patterns:
        print(f"  - {p['pattern']}: {p['frequency_ratio']:.1%} (contexts: {p['context_diversity']})")

    # Test LocalEntropyMicroscope
    print("\n2. Local Entropy Microscope")
    print("-" * 40)

    config = ToolConfig(tool_name="entropy_microscope")
    lem = LocalEntropyMicroscope(config, clock=clock)

    # Show LayeredTool architectural metadata
    arch_meta = lem.get_architectural_metadata()
    print(f"  Layer: {arch_meta['layer']} ({arch_meta['layer_name']})")
    print(f"  Phase: {arch_meta['phase']} ({arch_meta['phase_name']})")

    handle = registry.register_tool(lem)
    registry.lifecycle_manager.attach_tool(handle.tool_id)

    # Observe some regions
    import random
    regions = ["cache", "heap", "stack", "registers"]
    for _ in range(20):
        region = random.choice(regions)
        state = random.randint(0, 100)
        lem.observe_region(region, state)

    field = lem.get_entropy_field()
    print(f"Entropy field:")
    for region, entropy in field.items():
        print(f"  - {region}: {entropy:.2f} bits")

    # Test EventSignatureClassifier
    print("\n3. Event Signature Classifier")
    print("-" * 40)

    config = ToolConfig(tool_name="event_classifier")
    esc = EventSignatureClassifier(config, clock=clock)

    # Show LayeredTool architectural metadata
    arch_meta = esc.get_architectural_metadata()
    print(f"  Layer: {arch_meta['layer']} ({arch_meta['layer_name']})")
    print(f"  Phase: {arch_meta['phase']} ({arch_meta['phase_name']})")

    handle = registry.register_tool(esc)
    registry.lifecycle_manager.attach_tool(handle.tool_id)

    # Classify some events
    events = [
        {"type": "alloc", "size": 1024},
        {"type": "read", "fd": 0},
        {"type": "error", "exception": "ValueError"},
    ]

    print(f"Classification results:")
    for event in events:
        result = esc.classify_event(event)
        print(f"  - {event}: {result['classification']} (confidence: {result['confidence']:.1%})")

    stats = esc.get_classification_stats()
    print(f"\nClassification stats:")
    print(f"  Total: {stats['total']}")
    print(f"  By class: {stats['by_class']}")

    # Demonstrate hook overhead enforcement
    print("\n4. Hook Overhead Enforcement")
    print("-" * 40)

    hook_point = HookPoint(
        name="test_hook",
        timing=HookTiming.BEFORE,
        overhead_budget_ns=1_000_000,  # 1ms budget
        violation_threshold=3,  # Detach after 3 strikes
        clock=clock
    )

    # Add a slow callback that will violate overhead budget
    def slow_callback(ctx):
        clock.advance(0.002)  # 2ms (exceeds 1ms budget)
        return "slow"

    # Add a fast callback
    def fast_callback(ctx):
        clock.advance(0.0001)  # 0.1ms (within budget)
        return "fast"

    hook_point.register(slow_callback, priority=10)
    hook_point.register(fast_callback, priority=5)

    # Execute hook multiple times to trigger detachment
    import threading
    for i in range(5):
        ctx = HookContext(
            timestamp=clock.now(),
            cycle_count=i,
            thread_id=threading.get_ident()
        )
        hook_point.execute(ctx)
        clock.advance(0.1)

    # Check detached callbacks
    callback_stats = hook_point.get_callback_statistics()
    detached = [s for s in callback_stats if not s['enabled']]
    print(f"  Detached callbacks: {len(detached)}")
    if detached:
        for stat in detached:
            print(f"    - Callback {stat['callback_id']}: {stat['detachment_reason']}")
    print(f"  ✓ Automatic enforcement working!")

    # Demonstrate resource governor statistics
    print("\n5. Resource Governor Statistics")
    print("-" * 40)

    if registry._resource_governor:
        stats = registry._resource_governor.get_statistics()
        print(f"  Active tools: {stats['active_tools']}")
        print(f"  Total violations: {stats['total_violations']}")
        print(f"  Monitoring active: {stats['monitoring_active']}")
        print(f"  ✓ Resource governor operational!")

    print("\n✅ Demo complete!")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Void-State Tools System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  void-state --version           Show version information
  void-state status              Show deployment status
  void-state list                List all available tools
  void-state demo                Run a demo of MVP tools
  void-state --serve-metrics     Start metrics HTTP server
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Void-State v{__version__}"
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["status", "list", "demo"],
        help="Command to run"
    )

    parser.add_argument(
        "--serve-metrics",
        action="store_true",
        help="Start metrics HTTP server"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for metrics server (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for metrics server (default: 8000)"
    )

    args = parser.parse_args()

    # Handle --serve-metrics flag
    if args.serve_metrics:
        serve_metrics(args.host, args.port)
        return

    # Handle commands
    if args.command == "status":
        show_status()
    elif args.command == "list":
        list_tools()
    elif args.command == "demo":
        run_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
