#!/usr/bin/env python3
"""
Quick Start Example: Basic Tool Usage

A simple, practical introduction to Void-State tools for new users.
Demonstrates essential concepts without complexity:
- Tool initialization and configuration
- Basic analysis workflows
- Registry integration
- Statistics and monitoring

Perfect for learning the fundamentals before diving into advanced integrations.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from void_state_tools import (
    ToolRegistry,
    ToolConfig,
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier,
)
from void_state_tools.clock import DeterministicClock


def example_1_basic_tool_usage():
    """Example 1: Initialize and use a single tool."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Tool Usage")
    print("=" * 70 + "\n")

    # Step 1: Create a tool configuration
    config = ToolConfig(
        tool_name="my_pattern_analyzer",
        max_memory_mb=100,  # Memory limit
        max_cpu_percent=10,  # CPU limit
        overhead_budget_ns=1000,  # Performance budget
    )

    # Step 2: Initialize the tool
    tool = PatternPrevalenceQuantifier(config)

    print("✓ Tool initialized: PatternPrevalenceQuantifier")
    print(f"  Config: {config.tool_name}")
    print(f"  Layer: {tool.get_metadata()['layer']}")
    print(f"  Phase: {tool.get_metadata()['phase']}")

    # Step 3: Analyze some data
    result = tool.analyze({
        "pattern": "memory_allocation",
        "context": "production_server",
        "timestamp": 1234567890.0,
    })

    print(f"\n✓ Analysis complete")
    print(f"  Pattern: memory_allocation")
    print(f"  Frequency Ratio: {result['frequency_ratio']:.2%}")
    print(f"  Count: {result['frequency']}")
    print(f"  Context Diversity: {result['context_diversity']}")

    # Step 4: Check tool metadata
    metadata = tool.get_metadata()
    print(f"\n✓ Tool Metadata")
    print(f"  Tool Type: {metadata.get('tool_type', 'Unknown')}")
    print(f"  Complexity: {metadata.get('complexity', 'Unknown')}")


def example_2_registry_integration():
    """Example 2: Using the Tool Registry for lifecycle management."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Registry Integration")
    print("=" * 70 + "\n")

    # Step 1: Create registry
    registry = ToolRegistry()
    print("✓ Registry created")

    # Step 2: Create and register multiple tools
    tools = []

    # Pattern analyzer
    pattern_tool = PatternPrevalenceQuantifier(
        ToolConfig(tool_name="pattern_analyzer", max_memory_mb=100)
    )
    handle1 = registry.register_tool(pattern_tool)
    tools.append(("pattern_analyzer", handle1))

    # Entropy analyzer
    entropy_tool = LocalEntropyMicroscope(
        ToolConfig(tool_name="entropy_analyzer", max_memory_mb=100)
    )
    handle2 = registry.register_tool(entropy_tool)
    tools.append(("entropy_analyzer", handle2))

    # Event classifier
    classifier_tool = EventSignatureClassifier(
        ToolConfig(tool_name="event_classifier", max_memory_mb=100)
    )
    handle3 = registry.register_tool(classifier_tool)
    tools.append(("event_classifier", handle3))

    print(f"✓ Registered {len(tools)} tools")

    # Step 3: Activate tools through lifecycle manager
    for name, handle in tools:
        registry.lifecycle_manager.attach_tool(handle.tool_id)
        print(f"  → {name}: {handle.tool_id}")

    # Step 4: List all tools
    all_tools = registry.list_tools()
    print(f"\n✓ Active tools: {len(all_tools)}")

    # Step 5: List registered tools
    for i, tool_handle in enumerate(all_tools, 1):
        print(f"  {i}. {tool_handle.tool_id[:8]}... (active)")


def example_3_multi_tool_analysis():
    """Example 3: Analyzing multiple patterns with one tool."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Pattern Analysis Across Multiple Events")
    print("=" * 70 + "\n")

    # Initialize a single tool with deterministic clock
    clock = DeterministicClock()

    pattern_tool = PatternPrevalenceQuantifier(
        ToolConfig(tool_name="pattern", max_memory_mb=100),
        clock=clock,
    )

    print("✓ Initialized Pattern Analyzer\n")

    # Analyze multiple events
    events = [
        {"pattern": "login", "context": "web_app"},
        {"pattern": "logout", "context": "web_app"},
        {"pattern": "login", "context": "web_app"},
        {"pattern": "error", "context": "web_app"},
        {"pattern": "login", "context": "mobile_app"},
    ]

    print("Analyzing event sequence...")
    print("-" * 70)

    for i, event in enumerate(events, 1):
        result = pattern_tool.analyze({
            "pattern": event["pattern"],
            "context": event["context"],
            "timestamp": clock.now(),
        })
        clock.advance(1.0)

        print(f"\n{i}. {event['pattern']} from {event['context']}")
        print(f"   Frequency: {result['frequency_ratio']:.2%}")
        print(f"   Contexts: {result['context_diversity']}")

    # Summary
    print("\n" + "─" * 70)
    print("Summary:")
    print(f"  Total events analyzed: {pattern_tool.total_observations}")
    print(f"  Unique patterns: {len(pattern_tool.pattern_counts)}")
    print(f"\n  Pattern breakdown:")
    for pattern, count in sorted(pattern_tool.pattern_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / pattern_tool.total_observations) * 100
        print(f"    - {pattern}: {count} ({percentage:.0f}%)")


def example_4_tool_statistics():
    """Example 4: Monitoring tool performance and statistics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Tool Performance Monitoring")
    print("=" * 70 + "\n")

    clock = DeterministicClock()
    tool = PatternPrevalenceQuantifier(
        ToolConfig(tool_name="monitored_tool", max_memory_mb=100),
        clock=clock,
    )

    # Run multiple analyses
    patterns = ["login", "logout", "login", "error", "login", "warning"]

    print("Running analyses...")
    for i, pattern in enumerate(patterns):
        tool.analyze({
            "pattern": pattern,
            "context": "user_session",
            "timestamp": clock.now(),
        })
        clock.advance(1.0)
        print(f"  {i+1}. Analyzed pattern: {pattern}")

    # Access tool's internal tracking
    print(f"\n✓ Performance Statistics")
    print(f"  Total Observations: {tool.total_observations}")
    print(f"  Unique Patterns Tracked: {len(tool.pattern_counts)}")

    # Get pattern-specific data
    if tool.pattern_counts:
        print(f"\n  Pattern Breakdown:")
        for pattern, count in sorted(tool.pattern_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / tool.total_observations) * 100 if tool.total_observations > 0 else 0
            print(f"    - {pattern}: {count} ({percentage:.1f}%)")


def main():
    """Run all examples."""
    print("\n" + "═" * 70)
    print("VOID-STATE TOOLS: QUICK START GUIDE")
    print("═" * 70)

    print("\nThis guide demonstrates:")
    print("  1. Basic tool initialization and usage")
    print("  2. Tool registry and lifecycle management")
    print("  3. Multi-tool analysis pipelines")
    print("  4. Performance monitoring and statistics")

    # Run all examples
    example_1_basic_tool_usage()
    example_2_registry_integration()
    example_3_multi_tool_analysis()
    example_4_tool_statistics()

    print("\n" + "═" * 70)
    print("QUICK START COMPLETE")
    print("═" * 70)
    print("\nNext Steps:")
    print("  → Explore advanced examples:")
    print("    - security_monitoring_integration.py")
    print("    - adaptive_system_health_monitoring.py")
    print("    - meta_tooling_demo.py")
    print("  → Read documentation: VOID_STATE_TOOLS_README.md")
    print("  → Check API reference: API.md")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
