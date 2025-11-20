"""
Integration tests for ToolRegistry with all MVP tools.

Tests the complete lifecycle, hook integration, and overhead enforcement
for all Phase 1 MVP tools.
"""

import pytest
import time
from typing import Dict, Any

from void_state_tools import (
    ToolRegistry,
    ToolConfig,
    ToolState,
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier,
    HookPoint,
    HookContext,
    HookTiming,
)


class TestToolRegistry:
    """Test ToolRegistry functionality."""

    def test_registry_creation(self):
        """Test creating a tool registry."""
        registry = ToolRegistry()
        assert registry is not None
        assert len(registry.list_tools()) == 0

    def test_tool_registration(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        config = ToolConfig(tool_name="test_tool")
        tool = PatternPrevalenceQuantifier(config)

        handle = registry.register_tool(tool)
        assert handle is not None
        assert handle.state == ToolState.DORMANT

        # Verify tool is in registry
        assert len(registry.list_tools()) == 1
        assert registry.get_tool(handle.tool_id) == handle

    def test_tool_lifecycle(self):
        """Test complete tool lifecycle."""
        registry = ToolRegistry()
        config = ToolConfig(tool_name="lifecycle_test")
        tool = PatternPrevalenceQuantifier(config)

        # Register
        handle = registry.register_tool(tool)
        assert handle.state == ToolState.DORMANT

        # Attach (DORMANT -> INITIALIZING -> ACTIVE)
        registry.lifecycle_manager.attach_tool(handle.tool_id)
        assert handle.state == ToolState.ACTIVE

        # Suspend (ACTIVE -> SUSPENDED)
        registry.lifecycle_manager.suspend_tool(handle.tool_id)
        assert handle.state == ToolState.SUSPENDED

        # Resume (SUSPENDED -> ACTIVE)
        registry.lifecycle_manager.resume_tool(handle.tool_id)
        assert handle.state == ToolState.ACTIVE

        # Detach (ACTIVE -> TERMINATED)
        registry.lifecycle_manager.detach_tool(handle.tool_id)
        assert handle.state == ToolState.TERMINATED

    def test_multiple_tool_registration(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()

        tools = [
            PatternPrevalenceQuantifier(ToolConfig(tool_name="ppq")),
            LocalEntropyMicroscope(ToolConfig(tool_name="lem")),
            EventSignatureClassifier(ToolConfig(tool_name="esc")),
        ]

        handles = [registry.register_tool(tool) for tool in tools]
        assert len(handles) == 3
        assert len(registry.list_tools()) == 3

    def test_tool_discovery_by_category(self):
        """Test discovering tools by category."""
        registry = ToolRegistry()

        config1 = ToolConfig(tool_name="ppq", tool_category="prevalence")
        config2 = ToolConfig(tool_name="lem", tool_category="entropy")

        registry.register_tool(PatternPrevalenceQuantifier(config1))
        registry.register_tool(LocalEntropyMicroscope(config2))

        prevalence_tools = registry.list_tools(category="prevalence")
        assert len(prevalence_tools) == 1

        entropy_tools = registry.list_tools(category="entropy")
        assert len(entropy_tools) == 1


class TestPatternPrevalenceQuantifier:
    """Test PatternPrevalenceQuantifier MVP tool."""

    def test_initialization(self):
        """Test tool initialization."""
        config = ToolConfig(tool_name="ppq_test")
        tool = PatternPrevalenceQuantifier(config)

        assert tool.initialize()
        assert tool.total_observations == 0

    def test_pattern_analysis(self):
        """Test pattern analysis."""
        config = ToolConfig(tool_name="ppq_test")
        tool = PatternPrevalenceQuantifier(config)
        tool.initialize()

        # Analyze patterns
        result = tool.analyze({
            "pattern": "memory_spike",
            "context": "inference",
            "timestamp": time.time()
        })

        assert result["pattern"] == "memory_spike"
        assert result["frequency"] == 1
        assert 0 <= result["frequency_ratio"] <= 1
        assert result["context_diversity"] == 1

    def test_pattern_prevalence_tracking(self):
        """Test tracking pattern prevalence over time."""
        config = ToolConfig(tool_name="ppq_test")
        tool = PatternPrevalenceQuantifier(config)
        tool.initialize()

        # Add same pattern multiple times
        for i in range(10):
            tool.analyze({
                "pattern": "test_pattern",
                "context": f"context_{i % 3}",  # 3 different contexts
                "timestamp": time.time()
            })

        result = tool.analyze({
            "pattern": "test_pattern",
            "context": "context_1",
            "timestamp": time.time()
        })

        assert result["frequency"] == 11
        assert result["context_diversity"] == 3

    def test_top_patterns(self):
        """Test getting top patterns."""
        config = ToolConfig(tool_name="ppq_test")
        tool = PatternPrevalenceQuantifier(config)
        tool.initialize()

        # Add multiple patterns with different frequencies
        patterns = [
            ("pattern_a", 10),
            ("pattern_b", 5),
            ("pattern_c", 2),
        ]

        for pattern, count in patterns:
            for _ in range(count):
                tool.analyze({
                    "pattern": pattern,
                    "context": "test",
                    "timestamp": time.time()
                })

        top = tool.get_top_patterns(2)
        assert len(top) == 2
        assert top[0]["pattern"] == "pattern_a"
        assert top[0]["frequency"] == 10

    def test_rare_patterns(self):
        """Test identifying rare patterns."""
        config = ToolConfig(tool_name="ppq_test")
        tool = PatternPrevalenceQuantifier(config)
        tool.initialize()

        # Add common and rare patterns
        for _ in range(100):
            tool.analyze({"pattern": "common", "context": "test"})

        for _ in range(1):
            tool.analyze({"pattern": "rare", "context": "test"})

        rare = tool.get_rare_patterns(threshold=0.05)
        assert len(rare) == 1
        assert rare[0]["pattern"] == "rare"


class TestLocalEntropyMicroscope:
    """Test LocalEntropyMicroscope MVP tool."""

    def test_initialization(self):
        """Test tool initialization."""
        config = ToolConfig(tool_name="lem_test")
        tool = LocalEntropyMicroscope(config)

        assert tool.initialize()
        assert len(tool.region_states) == 0

    def test_region_observation(self):
        """Test observing region state."""
        config = ToolConfig(tool_name="lem_test")
        tool = LocalEntropyMicroscope(config)
        tool.initialize()

        result = tool.observe_region("cache", 42)

        assert result["region"] == "cache"
        assert result["entropy"] >= 0
        assert result["observation_count"] == 1

    def test_entropy_calculation(self):
        """Test entropy calculation."""
        config = ToolConfig(tool_name="lem_test")
        tool = LocalEntropyMicroscope(config)
        tool.initialize()

        # Observe same state multiple times (low entropy)
        for _ in range(10):
            tool.observe_region("region1", "constant")

        # Observe random states (high entropy)
        import random
        for i in range(10):
            tool.observe_region("region2", random.randint(0, 100))

        field = tool.get_entropy_field()

        # Region with constant state should have very low entropy
        assert field["region1"] < 0.1

        # Region with random states should have higher entropy
        assert field["region2"] > field["region1"]

    def test_entropy_gradient(self):
        """Test entropy gradient detection."""
        config = ToolConfig(tool_name="lem_test")
        tool = LocalEntropyMicroscope(config)
        tool.initialize()

        # Create gradient by changing from constant to variable
        for _ in range(10):
            tool.observe_region("region", "constant")

        import random
        for _ in range(10):
            tool.observe_region("region", random.randint(0, 100))

        result = tool.observe_region("region", random.randint(0, 100))

        # Gradient should be positive (increasing entropy)
        assert "gradient" in result

    def test_anomaly_detection(self):
        """Test entropy anomaly detection."""
        config = ToolConfig(tool_name="lem_test")
        tool = LocalEntropyMicroscope(config)
        tool.initialize()

        # Create regions with normal entropy
        import random
        for _ in range(20):
            tool.observe_region("normal1", random.randint(0, 10))
            tool.observe_region("normal2", random.randint(0, 10))

        # Create anomalous region with very high entropy
        for _ in range(20):
            tool.observe_region("anomaly", random.randint(0, 1000))

        # Use a lower threshold (0.5 bits) to detect entropy differences
        anomalies = tool.identify_anomalies(threshold=0.5)

        assert len(anomalies) > 0
        assert any(a["region"] == "anomaly" for a in anomalies)

    def test_monitoring_event_handling(self):
        """Test on_event monitoring method."""
        config = ToolConfig(tool_name="lem_test")
        tool = LocalEntropyMicroscope(config)
        tool.initialize()

        # Send monitoring event
        event = {
            "region": "test_region",
            "state": 123
        }

        tool.on_event(event)

        # Verify state was recorded
        assert "test_region" in tool.region_states
        assert 123 in tool.region_states["test_region"]


class TestEventSignatureClassifier:
    """Test EventSignatureClassifier MVP tool."""

    def test_initialization(self):
        """Test tool initialization."""
        config = ToolConfig(tool_name="esc_test")
        tool = EventSignatureClassifier(config)

        assert tool.initialize()
        assert tool._is_trained

    def test_event_classification(self):
        """Test classifying an event."""
        config = ToolConfig(tool_name="esc_test")
        tool = EventSignatureClassifier(config)
        tool.initialize()

        event = {"type": "alloc", "size": 1024}
        result = tool.classify_event(event)

        assert result["classification"] in tool.event_classes
        assert 0 <= result["confidence"] <= 1
        assert "probabilities" in result
        assert "features" in result

    def test_training(self):
        """Test training the classifier."""
        config = ToolConfig(tool_name="esc_test")
        tool = EventSignatureClassifier(config)

        training_data = [
            ({"type": "custom", "action": "foo"}, "custom_class"),
            ({"type": "custom", "action": "foo"}, "custom_class"),
            ({"type": "other", "action": "bar"}, "other_class"),
        ]

        tool.train(training_data)

        assert tool._is_trained
        assert tool.training_count == 3

        # Classify with trained model
        result = tool.classify_event({"type": "custom", "action": "foo"})
        assert result["classification"] == "custom_class"

    def test_feature_extraction(self):
        """Test feature extraction."""
        config = ToolConfig(tool_name="esc_test")
        tool = EventSignatureClassifier(config)

        event = {
            "type": "alloc",
            "size": 50,
            "address": "0x1234"
        }

        features = tool._extract_features(event)

        assert "type:alloc" in features
        assert "has_size" in features
        assert "has_address" in features
        assert "size_small" in features  # size < 100

    def test_classification_stats(self):
        """Test classification statistics."""
        config = ToolConfig(tool_name="esc_test")
        tool = EventSignatureClassifier(config)
        tool.initialize()

        # Classify multiple events
        events = [
            {"type": "alloc", "size": 100},
            {"type": "alloc", "size": 200},
            {"type": "read", "fd": 0},
        ]

        for event in events:
            tool.classify_event(event)

        stats = tool.get_classification_stats()

        assert stats["total"] == 3
        assert "by_class" in stats
        assert "average_confidence" in stats
        assert stats["model_trained"]


class TestHookIntegration:
    """Test hook integration with tools."""

    def test_hook_registration(self):
        """Test registering hooks."""
        hook = HookPoint("test.hook", HookTiming.BEFORE, overhead_budget_ns=1000)

        call_count = [0]

        def callback(context, *args, **kwargs):
            call_count[0] += 1

        reg_id = hook.register(callback)
        assert reg_id is not None

        # Execute hook
        context = HookContext(
            timestamp=time.time(),
            cycle_count=1,
            thread_id=0,
            additional_data={}
        )

        hook.execute(context)
        assert call_count[0] == 1

    def test_hook_overhead_budget(self):
        """Test hook overhead budget enforcement."""
        hook = HookPoint("test.hook", HookTiming.BEFORE, overhead_budget_ns=100)

        def slow_callback(context, *args, **kwargs):
            time.sleep(0.001)  # 1ms = 1,000,000 ns (way over budget)

        hook.register(slow_callback)

        context = HookContext(
            timestamp=time.time(),
            cycle_count=1,
            thread_id=0,
            additional_data={}
        )

        # This should trigger budget warning
        hook.execute(context)

    def test_hook_priority_ordering(self):
        """Test hook execution priority ordering."""
        hook = HookPoint("test.hook", HookTiming.BEFORE)

        execution_order = []

        def high_priority(context):
            execution_order.append("high")

        def low_priority(context):
            execution_order.append("low")

        hook.register(high_priority, priority=100)
        hook.register(low_priority, priority=10)

        context = HookContext(
            timestamp=time.time(),
            cycle_count=1,
            thread_id=0,
            additional_data={}
        )

        hook.execute(context)

        # High priority should execute first
        assert execution_order == ["high", "low"]


class TestIntegrationScenario:
    """Test complete integration scenarios."""

    def test_full_mvp_integration(self):
        """Test all MVP tools working together."""
        # Create registry
        registry = ToolRegistry()

        # Create and register all MVP tools
        tools_config = [
            (PatternPrevalenceQuantifier, "ppq", "prevalence_novelty_quantifiers"),
            (LocalEntropyMicroscope, "lem", "entropy_zeal_microscopics"),
            (EventSignatureClassifier, "esc", "anomaly_event_classifiers"),
        ]

        handles = []
        for tool_cls, name, category in tools_config:
            config = ToolConfig(
                tool_name=name,
                tool_category=category,
                max_memory_mb=100,
                max_cpu_percent=10,
                overhead_budget_ns=1000
            )

            tool = tool_cls(config)
            handle = registry.register_tool(tool)
            registry.lifecycle_manager.attach_tool(handle.tool_id)
            handles.append(handle)

        # Verify all tools are active
        assert all(h.state == ToolState.ACTIVE for h in handles)
        assert len(registry.list_tools(state=ToolState.ACTIVE)) == 3

        # Test each tool
        ppq = handles[0]._tool
        lem = handles[1]._tool
        esc = handles[2]._tool

        # PatternPrevalenceQuantifier
        ppq_result = ppq.analyze({
            "pattern": "integration_test",
            "context": "test_suite",
            "timestamp": time.time()
        })
        assert ppq_result["pattern"] == "integration_test"

        # LocalEntropyMicroscope
        lem_result = lem.observe_region("integration_region", 42)
        assert lem_result["region"] == "integration_region"

        # EventSignatureClassifier
        esc_result = esc.classify_event({"type": "alloc", "size": 256})
        assert esc_result["classification"] == "memory_allocation"

        # Cleanup - detach all tools
        for handle in handles:
            registry.lifecycle_manager.detach_tool(handle.tool_id)

        assert all(h.state == ToolState.TERMINATED for h in handles)

    def test_synthetic_hook_scenario(self):
        """Test synthetic hook firing with tools."""
        registry = ToolRegistry()

        # Create tool
        config = ToolConfig(tool_name="test_tool")
        tool = LocalEntropyMicroscope(config)
        handle = registry.register_tool(tool)
        registry.lifecycle_manager.attach_tool(handle.tool_id)

        # Create synthetic hook with reasonable overhead budget (10ms for Python callbacks)
        hook = HookPoint("vm.before_cycle", HookTiming.BEFORE, overhead_budget_ns=10_000_000)

        # Register tool callback
        def tool_callback(context):
            # Simulate tool observing state on each cycle
            tool.observe_region("vm_state", context.cycle_count % 10)

        hook.register(tool_callback, priority=50)

        # Fire synthetic hooks
        for cycle in range(100):
            context = HookContext(
                timestamp=time.time(),
                cycle_count=cycle,
                thread_id=0,
                additional_data={}
            )
            hook.execute(context)

        # Verify tool collected data
        assert "vm_state" in tool.region_states
        assert len(tool.region_states["vm_state"]) == 100

        # Verify entropy was calculated
        field = tool.get_entropy_field()
        assert "vm_state" in field
        assert field["vm_state"] > 0  # Should have some entropy

    def test_resource_overhead_budget(self):
        """Test that tools respect overhead budgets."""
        registry = ToolRegistry()

        # Create tool with strict budget
        config = ToolConfig(
            tool_name="test_tool",
            overhead_budget_ns=100  # Very strict: 100ns
        )

        tool = PatternPrevalenceQuantifier(config)
        handle = registry.register_tool(tool)
        registry.lifecycle_manager.attach_tool(handle.tool_id)

        # Verify tool is active
        assert handle.state == ToolState.ACTIVE

        # Simulate repeated operations
        for i in range(10):
            tool.analyze({
                "pattern": f"pattern_{i % 3}",
                "context": "test",
                "timestamp": time.time()
            })

        # Tool should still be active (analysis is fast enough)
        assert handle.state == ToolState.ACTIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
