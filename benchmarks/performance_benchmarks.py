#!/usr/bin/env python3
"""
Performance Benchmarks for Void-State Tools

Comprehensive benchmark suite measuring:
- Execution time (overhead)
- Memory usage
- Throughput (operations/second)
- Scalability (with varying input sizes)

All benchmarks use deterministic clocks for reproducibility.
"""

import sys
import time
import gc
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from void_state_tools import (
    ToolConfig,
    # Phase 1
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier,
    # Phase 2
    ThreatSignatureRecognizer,
    BehavioralAnomalyDetector,
    NoveltyDetector,
    EmergentPatternRecognizer,
    TimelineBranchingEngine,
    ProphecyEngine,
    ObserverEffectDetector,
    CausalInterventionSimulator,
)
from void_state_tools.clock import DeterministicClock


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    tool_name: str
    operations: int
    total_time_ms: float
    avg_time_us: float  # microseconds
    ops_per_second: float
    memory_peak_mb: float
    memory_delta_mb: float
    overhead_target_met: bool
    overhead_target_us: float


class PerformanceBenchmark:
    """Performance benchmark harness."""

    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []

    def benchmark_tool(
        self,
        tool_name: str,
        tool_factory: Callable,
        operation: Callable,
        overhead_target_us: float = 1000.0,  # 1ms default
    ) -> BenchmarkResult:
        """
        Benchmark a tool's performance.

        Args:
            tool_name: Name of the tool
            tool_factory: Function that creates tool instance
            operation: Function that performs one operation on the tool
            overhead_target_us: Target overhead in microseconds

        Returns:
            BenchmarkResult with performance metrics
        """
        # Create tool instance
        tool = tool_factory()

        # Warmup
        for _ in range(self.warmup_runs):
            operation(tool)

        # Start memory tracking
        gc.collect()
        tracemalloc.start()
        memory_before = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

        # Benchmark
        start_time = time.perf_counter()

        for _ in range(self.benchmark_runs):
            operation(tool)

        end_time = time.perf_counter()

        # End memory tracking
        memory_after_current, memory_after_peak = tracemalloc.get_traced_memory()
        memory_after_current_mb = memory_after_current / 1024 / 1024
        memory_after_peak_mb = memory_after_peak / 1024 / 1024
        tracemalloc.stop()

        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        avg_time_us = (total_time_ms * 1000) / self.benchmark_runs
        ops_per_second = self.benchmark_runs / (total_time_ms / 1000)
        memory_delta_mb = memory_after_current_mb - memory_before

        result = BenchmarkResult(
            tool_name=tool_name,
            operations=self.benchmark_runs,
            total_time_ms=total_time_ms,
            avg_time_us=avg_time_us,
            ops_per_second=ops_per_second,
            memory_peak_mb=memory_after_peak_mb,
            memory_delta_mb=memory_delta_mb,
            overhead_target_met=avg_time_us <= overhead_target_us,
            overhead_target_us=overhead_target_us,
        )

        self.results.append(result)
        return result

    def print_results(self):
        """Print formatted benchmark results."""
        print("\n" + "=" * 100)
        print("VOID-STATE TOOLS: PERFORMANCE BENCHMARKS")
        print("=" * 100)
        print(f"\nBenchmark Configuration:")
        print(f"  Warmup runs: {self.warmup_runs}")
        print(f"  Benchmark runs: {self.benchmark_runs}")
        print()

        # Group by phase
        phase1_tools = [r for r in self.results if r.tool_name in [
            "PatternPrevalenceQuantifier", "LocalEntropyMicroscope", "EventSignatureClassifier"
        ]]
        phase2_tools = [r for r in self.results if r.tool_name not in [
            "PatternPrevalenceQuantifier", "LocalEntropyMicroscope", "EventSignatureClassifier"
        ]]

        self._print_phase_results("Phase 1 Tools (MVP)", phase1_tools)
        self._print_phase_results("Phase 2 Tools (Growth)", phase2_tools)

        # Summary
        print("\n" + "-" * 100)
        print("SUMMARY")
        print("-" * 100)

        total_tools = len(self.results)
        tools_meeting_target = sum(1 for r in self.results if r.overhead_target_met)
        avg_overhead = sum(r.avg_time_us for r in self.results) / total_tools

        print(f"  Total Tools Benchmarked: {total_tools}")
        print(f"  Tools Meeting Overhead Target: {tools_meeting_target}/{total_tools} ({tools_meeting_target/total_tools*100:.0f}%)")
        print(f"  Average Overhead: {avg_overhead:.2f}µs")
        print(f"  Fastest Tool: {min(self.results, key=lambda r: r.avg_time_us).tool_name} ({min(r.avg_time_us for r in self.results):.2f}µs)")
        print(f"  Slowest Tool: {max(self.results, key=lambda r: r.avg_time_us).tool_name} ({max(r.avg_time_us for r in self.results):.2f}µs)")
        print("\n" + "=" * 100)

    def _print_phase_results(self, phase_name: str, results: List[BenchmarkResult]):
        """Print results for a specific phase."""
        if not results:
            return

        print(f"\n{phase_name}")
        print("-" * 100)
        print(f"{'Tool':<35} {'Ops':<8} {'Avg Time':<12} {'Ops/sec':<12} {'Mem Δ':<10} {'Target Met':<12}")
        print("-" * 100)

        for result in results:
            status = "✓ YES" if result.overhead_target_met else "✗ NO"
            target_info = f"({result.overhead_target_us:.0f}µs)"

            print(f"{result.tool_name:<35} "
                  f"{result.operations:<8} "
                  f"{result.avg_time_us:>8.2f}µs   "
                  f"{result.ops_per_second:>9.0f}   "
                  f"{result.memory_delta_mb:>6.2f}MB  "
                  f"{status} {target_info}")


def run_phase1_benchmarks(harness: PerformanceBenchmark):
    """Benchmark Phase 1 tools."""
    print("\nBenchmarking Phase 1 Tools...")

    clock = DeterministicClock()

    # PatternPrevalenceQuantifier
    def create_pattern_tool():
        return PatternPrevalenceQuantifier(
            ToolConfig(tool_name="benchmark", max_memory_mb=100),
            clock=clock
        )

    def pattern_operation(tool):
        tool.analyze({
            "pattern": "test_pattern",
            "context": "benchmark",
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "PatternPrevalenceQuantifier",
        create_pattern_tool,
        pattern_operation,
        overhead_target_us=1000,  # 1ms
    )

    # LocalEntropyMicroscope
    def create_entropy_tool():
        return LocalEntropyMicroscope(
            ToolConfig(tool_name="benchmark", max_memory_mb=100),
            clock=clock
        )

    def entropy_operation(tool):
        tool.observe_region("test_region", "sample_data_" * 10)
        clock.advance(0.001)

    harness.benchmark_tool(
        "LocalEntropyMicroscope",
        create_entropy_tool,
        entropy_operation,
        overhead_target_us=1000,  # 1ms
    )

    # EventSignatureClassifier
    def create_classifier_tool():
        tool = EventSignatureClassifier(
            ToolConfig(tool_name="benchmark", max_memory_mb=100),
            clock=clock
        )
        # Train with sample data
        tool.train([
            ({"features": ["login", "success"]}, "authentication"),
            ({"features": ["error", "timeout"]}, "failure"),
        ])
        return tool

    def classifier_operation(tool):
        tool.classify_event({
            "features": ["login", "success"],
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "EventSignatureClassifier",
        create_classifier_tool,
        classifier_operation,
        overhead_target_us=2000,  # 2ms
    )


def run_phase2_benchmarks(harness: PerformanceBenchmark):
    """Benchmark Phase 2 tools."""
    print("\nBenchmarking Phase 2 Tools...")

    clock = DeterministicClock()

    # ThreatSignatureRecognizer
    def create_threat_tool():
        return ThreatSignatureRecognizer(
            ToolConfig(tool_name="benchmark", max_memory_mb=200),
            clock=clock
        )

    def threat_operation(tool):
        tool.analyze({
            "event_data": {"type": "network", "payload": "test"},
            "context": "benchmark",
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "ThreatSignatureRecognizer",
        create_threat_tool,
        threat_operation,
        overhead_target_us=5000,  # 5ms
    )

    # BehavioralAnomalyDetector
    from void_state_tools import BehaviorTrace, BehaviorProfile

    def create_anomaly_tool():
        tool = BehavioralAnomalyDetector(
            ToolConfig(tool_name="benchmark", max_memory_mb=150),
            clock=clock
        )
        # Learn a baseline profile
        baseline_trace = BehaviorTrace(
            actions=["login", "read", "write", "logout"],
            states=[{"session": "active"}, {"session": "active"}, {"session": "active"}, {"session": "closed"}],
            timestamps=[0.0, 1.0, 2.0, 3.0],
        )
        tool.learn_profile(baseline_trace)
        return tool

    def anomaly_operation(tool):
        # Create test trace
        test_trace = BehaviorTrace(
            actions=["login", "read", "logout"],
            states=[{"session": "active"}, {"session": "active"}, {"session": "closed"}],
            timestamps=[clock.now(), clock.now() + 1, clock.now() + 2],
        )
        # Create a basic profile for comparison
        profile = BehaviorProfile(
            expected_sequences={("login", "read", "logout")},
            state_transitions={},
            frequency_model={"login": 0.33, "read": 0.33, "logout": 0.34},
            temporal_patterns={},
        )
        tool.detect_anomaly(test_trace, profile)
        clock.advance(0.001)

    harness.benchmark_tool(
        "BehavioralAnomalyDetector",
        create_anomaly_tool,
        anomaly_operation,
        overhead_target_us=3000,  # 3ms
    )

    # NoveltyDetector
    def create_novelty_tool():
        return NoveltyDetector(
            ToolConfig(tool_name="benchmark", max_memory_mb=150),
            clock=clock
        )

    def novelty_operation(tool):
        tool.analyze({
            "observation": {"feature1": "value1"},
            "domain": "benchmark",
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "NoveltyDetector",
        create_novelty_tool,
        novelty_operation,
        overhead_target_us=200,  # 200µs
    )

    # EmergentPatternRecognizer
    def create_emergent_tool():
        return EmergentPatternRecognizer(
            ToolConfig(tool_name="benchmark", max_memory_mb=150),
            clock=clock
        )

    def emergent_operation(tool):
        tool.analyze({
            "observations": [("pattern_a", {}), ("pattern_b", {})],
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "EmergentPatternRecognizer",
        create_emergent_tool,
        emergent_operation,
        overhead_target_us=500,  # 500µs
    )

    # TimelineBranchingEngine
    def create_timeline_tool():
        return TimelineBranchingEngine(
            ToolConfig(tool_name="benchmark", max_memory_mb=200),
            clock=clock
        )

    def timeline_operation(tool):
        tool.analyze({
            "initial_state": {"x": 1},
            "fork_name": "test_fork",
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "TimelineBranchingEngine",
        create_timeline_tool,
        timeline_operation,
        overhead_target_us=5000,  # 5ms
    )

    # ProphecyEngine
    def create_prophecy_tool():
        return ProphecyEngine(
            ToolConfig(tool_name="benchmark", max_memory_mb=300),
            clock=clock
        )

    def prophecy_operation(tool):
        def simple_dynamics(state, dt):
            return {"x": state.get("x", 0) + dt}

        tool.analyze({
            "initial_state": {"x": 0},
            "dynamics_function": simple_dynamics,
            "time_horizon": 10.0,
            "num_trajectories": 10,
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "ProphecyEngine",
        create_prophecy_tool,
        prophecy_operation,
        overhead_target_us=10000,  # 10ms
    )

    # ObserverEffectDetector
    def create_observer_tool():
        return ObserverEffectDetector(
            ToolConfig(tool_name="benchmark", max_memory_mb=150),
            clock=clock
        )

    def observer_operation(tool):
        tool.analyze({
            "entity_id": "entity_1",
            "baseline_behavior": {"metric1": 100},
            "observed_behavior": {"metric1": 105},
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "ObserverEffectDetector",
        create_observer_tool,
        observer_operation,
        overhead_target_us=200,  # 200µs
    )

    # CausalInterventionSimulator
    def create_causal_tool():
        tool = CausalInterventionSimulator(
            ToolConfig(tool_name="benchmark", max_memory_mb=200),
            clock=clock
        )
        # Learn some structure
        tool.learn_causal_structure([
            {"x": 1, "y": 2},
            {"x": 2, "y": 4},
        ])
        return tool

    def causal_operation(tool):
        tool.analyze({
            "intervention": {"x": 5},
            "query_variables": ["y"],
            "timestamp": clock.now(),
        })
        clock.advance(0.001)

    harness.benchmark_tool(
        "CausalInterventionSimulator",
        create_causal_tool,
        causal_operation,
        overhead_target_us=1000,  # 1ms
    )


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 100)
    print("Starting Performance Benchmarks")
    print("=" * 100)

    # Create benchmark harness
    harness = PerformanceBenchmark(warmup_runs=10, benchmark_runs=1000)

    # Run benchmarks
    run_phase1_benchmarks(harness)
    run_phase2_benchmarks(harness)

    # Print results
    harness.print_results()

    print("\n📝 Note: Benchmarks run with deterministic clocks for reproducibility.")
    print("   Real-world performance may vary based on system load and hardware.")


if __name__ == "__main__":
    main()
