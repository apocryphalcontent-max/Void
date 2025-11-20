"""
Integration tests for Void-State Tools MVP.

Tests the integration of all core components:
- ToolRegistry
- HookRegistry  
- All three MVP tools (PatternPrevalenceQuantifier, LocalEntropyMicroscope, EventSignatureClassifier)
- Hook execution and overhead budgets
"""

import time

import pytest

from void_state_tools import (
    EventSignatureClassifier,
    HookContext,
    HookRegistry,
    LocalEntropyMicroscope,
    PatternPrevalenceQuantifier,
    ToolConfig,
    ToolRegistry,
    ToolState,
)


class TestToolRegistryIntegration:
    """Test ToolRegistry with MVP tools."""

    def test_registry_with_all_mvp_tools(self):
        """Test that all MVP tools can be registered and attached."""
        registry = ToolRegistry()

        # Create configurations for all tools
        tools = {
            "pattern_quantifier": PatternPrevalenceQuantifier(
                ToolConfig(tool_name="pattern_quantifier", overhead_budget_ns=1000)
            ),
            "entropy_microscope": LocalEntropyMicroscope(
                ToolConfig(tool_name="entropy_microscope", overhead_budget_ns=1000)
            ),
            "event_classifier": EventSignatureClassifier(
                ToolConfig(tool_name="event_classifier", overhead_budget_ns=1000)
            ),
        }

        # Register all tools
        handles = {}
        for name, tool in tools.items():
            handle = registry.register_tool(tool)
            handles[name] = handle
            assert handle is not None
            assert handle.state == ToolState.DORMANT

        # Attach all tools
        for name, handle in handles.items():
            success = registry.lifecycle_manager.attach_tool(handle.tool_id)
            assert success, f"Failed to attach {name}"
            assert handle.state == ToolState.ACTIVE

        # Verify all tools are active
        active_tools = registry.list_tools(state=ToolState.ACTIVE)
        assert len(active_tools) == 3

        # Detach all tools
        for name, handle in handles.items():
            success = registry.lifecycle_manager.detach_tool(handle.tool_id)
            assert success, f"Failed to detach {name}"
            assert handle.state == ToolState.TERMINATED

    def test_pattern_quantifier_integration(self):
        """Test PatternPrevalenceQuantifier with realistic data."""
        registry = ToolRegistry()

        config = ToolConfig(
            tool_name="pattern_test",
            max_memory_mb=50,
            overhead_budget_ns=1000
        )
        quantifier = PatternPrevalenceQuantifier(config)
        handle = registry.register_tool(quantifier)

        # Attach and verify
        registry.lifecycle_manager.attach_tool(handle.tool_id)
        assert handle.state == ToolState.ACTIVE

        # Analyze patterns
        patterns = [
            ("memory_alloc", "gc_cycle"),
            ("memory_alloc", "user_code"),
            ("io_read", "network"),
            ("memory_alloc", "gc_cycle"),  # Duplicate
        ]

        for pattern, context in patterns:
            result = quantifier.analyze({
                "pattern": pattern,
                "context": context,
                "timestamp": time.time()
            })
            assert "frequency" in result
            assert "context_diversity" in result

        # Verify statistics
        top_patterns = quantifier.get_top_patterns(n=5)
        assert len(top_patterns) > 0
        assert top_patterns[0]["pattern"] == "memory_alloc"
        assert top_patterns[0]["frequency"] == 3

        # Check metrics
        metrics = handle.metrics
        assert metrics.events_processed >= 0

        # Cleanup
        registry.lifecycle_manager.detach_tool(handle.tool_id)

    def test_entropy_microscope_integration(self):
        """Test LocalEntropyMicroscope with multiple regions."""
        registry = ToolRegistry()

        config = ToolConfig(
            tool_name="entropy_test",
            overhead_budget_ns=1000
        )
        microscope = LocalEntropyMicroscope(config)
        handle = registry.register_tool(microscope)

        # Attach
        registry.lifecycle_manager.attach_tool(handle.tool_id)
        assert handle.state == ToolState.ACTIVE

        # Observe multiple regions
        regions = ["heap", "stack", "static"]
        states = ["idle", "active", "idle", "active", "busy"]

        for region in regions:
            for state in states:
                result = microscope.observe_region(region, state)
                assert "entropy" in result
                assert "gradient" in result

        # Check entropy field
        field = microscope.get_entropy_field()
        assert len(field) == 3  # Three regions
        assert all(entropy >= 0 for entropy in field.values())

        # Identify anomalies
        anomalies = microscope.identify_anomalies(threshold=1.0)
        # Anomalies may or may not exist depending on the data

        # Cleanup
        registry.lifecycle_manager.detach_tool(handle.tool_id)

    def test_event_classifier_integration(self):
        """Test EventSignatureClassifier with training and classification."""
        registry = ToolRegistry()

        config = ToolConfig(
            tool_name="classifier_test",
            overhead_budget_ns=1000
        )
        classifier = EventSignatureClassifier(config)
        handle = registry.register_tool(classifier)

        # Attach
        registry.lifecycle_manager.attach_tool(handle.tool_id)
        assert handle.state == ToolState.ACTIVE

        # Train the classifier
        training_data = [
            ({"type": "memory_allocation", "size": 1024}, "memory_allocation"),
            ({"type": "memory_allocation", "size": 2048}, "memory_allocation"),
            ({"type": "io_read", "size": 512, "duration": 0.001}, "io_read"),
            ({"type": "io_write", "size": 1024, "duration": 0.002}, "io_write"),
            ({"type": "exception", "error": "NullPointer"}, "exception"),
        ]

        for event, true_class in training_data:
            classifier.train(event, true_class)

        # Classify events
        test_events = [
            {"type": "memory_allocation", "size": 3072},
            {"type": "io_read", "size": 256, "duration": 0.0015},
            {"type": "unknown_event", "data": "something"},
        ]

        for event in test_events:
            result = classifier.classify_event(event)
            assert "classification" in result
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0

        # Check statistics
        stats = classifier.get_classification_stats()
        assert stats["total_predictions"] == 3
        assert stats["total_training_samples"] == 5

        # Cleanup
        registry.lifecycle_manager.detach_tool(handle.tool_id)


class TestHookIntegration:
    """Test hook system with tools."""

    def test_hooks_with_synthetic_data(self):
        """Test firing synthetic hooks with tool callbacks."""
        hook_registry = HookRegistry()

        # Get VM hooks
        before_cycle = hook_registry.vm_hooks.before_cycle
        after_cycle = hook_registry.vm_hooks.after_cycle

        # Register callbacks
        call_counts = {"before": 0, "after": 0}

        def before_callback(context):
            call_counts["before"] += 1
            return f"before_{call_counts['before']}"

        def after_callback(context):
            call_counts["after"] += 1
            return f"after_{call_counts['after']}"

        before_cycle.register(before_callback)
        after_cycle.register(after_callback)

        # Fire hooks
        for i in range(5):
            context = HookContext(
                timestamp=time.time(),
                cycle_count=i,
                thread_id=0,
                additional_data={"cycle": i}
            )

            before_results = before_cycle.execute(context)
            after_results = after_cycle.execute(context)

            assert len(before_results) == 1
            assert len(after_results) == 1

        assert call_counts["before"] == 5
        assert call_counts["after"] == 5

    def test_hook_overhead_budget(self):
        """Test that hook overhead is measured and reported."""
        hook_registry = HookRegistry()
        hook = hook_registry.vm_hooks.before_cycle

        # Register a callback that takes some time
        def slow_callback(context):
            # Simulate some work (but not too much to avoid test slowness)
            time.sleep(0.0001)  # 100 microseconds
            return "slow"

        hook.register(slow_callback)

        # Execute hook and check overhead
        context = HookContext(
            timestamp=time.time(),
            cycle_count=0,
            thread_id=0,
            additional_data={}
        )

        start = time.perf_counter_ns()
        results = hook.execute(context)
        elapsed = time.perf_counter_ns() - start

        assert len(results) == 1
        assert elapsed > 0  # Some time was taken

        # The overhead budget is 100ns by default, but our callback takes 100us
        # so it will exceed the budget and print a warning


class TestToolMetrics:
    """Test tool metrics collection."""

    def test_metrics_collection(self):
        """Test that metrics are collected during tool operation."""
        registry = ToolRegistry()

        config = ToolConfig(tool_name="metrics_test")
        quantifier = PatternPrevalenceQuantifier(config)
        handle = registry.register_tool(quantifier)

        # Attach tool
        registry.lifecycle_manager.attach_tool(handle.tool_id)

        # Perform operations
        for i in range(10):
            quantifier.analyze({
                "pattern": f"pattern_{i % 3}",
                "context": "test",
                "timestamp": time.time()
            })

        # Check metrics
        metrics = handle.metrics
        assert metrics.total_runtime_seconds > 0

        # Detach
        registry.lifecycle_manager.detach_tool(handle.tool_id)


class TestToolSuspendResume:
    """Test tool suspension and resumption."""

    def test_suspend_and_resume(self):
        """Test that tools can be suspended and resumed."""
        registry = ToolRegistry()

        config = ToolConfig(tool_name="suspend_test")
        microscope = LocalEntropyMicroscope(config)
        handle = registry.register_tool(microscope)

        # Attach
        registry.lifecycle_manager.attach_tool(handle.tool_id)
        assert handle.state == ToolState.ACTIVE

        # Observe some data
        microscope.observe_region("region1", "state1")

        # Suspend
        success = registry.lifecycle_manager.suspend_tool(handle.tool_id)
        assert success
        assert handle.state == ToolState.SUSPENDED

        # Resume
        success = registry.lifecycle_manager.resume_tool(handle.tool_id)
        assert success
        assert handle.state == ToolState.ACTIVE

        # Verify data is still there
        field = microscope.get_entropy_field()
        assert "region1" in field

        # Detach
        registry.lifecycle_manager.detach_tool(handle.tool_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
