"""
Comprehensive tests for all Phase 1 enhancements.

Tests:
- Clock abstraction integration
- Hook overhead enforcement with automatic detachment
- ResourceGovernor integration with ToolRegistry
- LayeredTool mixin validation
"""

import pytest
import threading
import time
from void_state_tools import (
    ToolConfig,
    ToolRegistry,
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier,
    HookPoint,
    HookContext,
    HookTiming,
)
from void_state_tools.clock import DeterministicClock, SystemClock, get_clock, set_clock
from void_state_tools.resource_governor import (
    ResourceGovernor,
    QuotaPolicy,
    ResourceUsage,
    EnforcementAction,
)
from void_state_tools.layered_tool import LayerPhaseMismatchError


class TestClockAbstraction:
    """Test clock abstraction integration."""

    def test_deterministic_clock_in_tools(self):
        """Test that tools use injected deterministic clock."""
        clock = DeterministicClock(start_time=1000.0)
        config = ToolConfig(tool_name="test")

        # Test PatternPrevalenceQuantifier
        ppq = PatternPrevalenceQuantifier(config, clock=clock)

        # Analyze a pattern at time 1000
        result1 = ppq.analyze({
            "pattern": "test_pattern",
            "context": "test_context",
            "timestamp": clock.now()
        })
        assert result1["age_seconds"] == 0.0  # First observation

        # Advance clock and analyze again
        clock.advance(5.0)
        result2 = ppq.analyze({
            "pattern": "test_pattern",
            "context": "test_context",
            "timestamp": clock.now()
        })

        # Age should be 5 seconds now
        assert result2["age_seconds"] == 5.0

    def test_deterministic_clock_in_registry(self):
        """Test that ToolRegistry uses injected clock."""
        clock = DeterministicClock(start_time=1000.0)
        registry = ToolRegistry(clock=clock)

        config = ToolConfig(tool_name="test")
        tool = PatternPrevalenceQuantifier(config, clock=clock)
        handle = registry.register_tool(tool)

        # Advance time
        clock.advance(10.0)

        # Tool handle should use the same clock
        assert handle._clock == clock

    def test_system_clock_default(self):
        """Test that SystemClock is used by default."""
        config = ToolConfig(tool_name="test")
        tool = PatternPrevalenceQuantifier(config)

        # Should use global clock (SystemClock by default)
        assert tool._clock == get_clock()

    def test_deterministic_clock_sleep(self):
        """Test deterministic clock sleep advances time without blocking."""
        clock = DeterministicClock(start_time=0.0)

        start = clock.now()
        clock.sleep(5.0)
        end = clock.now()

        assert end - start == 5.0
        assert clock.get_sleep_count() == 1


class TestHookOverheadEnforcement:
    """Test hook overhead enforcement with automatic detachment."""

    def test_slow_callback_detachment(self):
        """Test that slow callbacks are automatically detached."""
        clock = DeterministicClock(start_time=1000.0)

        hook_point = HookPoint(
            name="test_hook",
            timing=HookTiming.BEFORE,
            overhead_budget_ns=1_000_000,  # 1ms budget
            violation_threshold=3,  # Detach after 3 strikes
            clock=clock
        )

        # Slow callback that exceeds budget
        def slow_callback(ctx):
            clock.advance(0.002)  # 2ms
            return "slow"

        hook_point.register(slow_callback, priority=10)

        # Execute hook 5 times (should detach after 3)
        for i in range(5):
            ctx = HookContext(
                timestamp=clock.now(),
                cycle_count=i,
                thread_id=threading.get_ident()
            )
            hook_point.execute(ctx)

        # Check that callback was detached
        stats = hook_point.get_callback_statistics()
        assert len(stats) == 1
        assert stats[0]['enabled'] is False
        assert stats[0]['detachment_reason'] is not None
        assert 'budget' in stats[0]['detachment_reason'].lower()

    def test_fast_callback_not_detached(self):
        """Test that fast callbacks are not detached."""
        clock = DeterministicClock(start_time=1000.0)

        hook_point = HookPoint(
            name="test_hook",
            timing=HookTiming.BEFORE,
            overhead_budget_ns=1_000_000,  # 1ms budget
            violation_threshold=3,
            clock=clock
        )

        # Fast callback within budget
        def fast_callback(ctx):
            clock.advance(0.0005)  # 0.5ms
            return "fast"

        hook_point.register(fast_callback, priority=10)

        # Execute hook 10 times
        for i in range(10):
            ctx = HookContext(
                timestamp=clock.now(),
                cycle_count=i,
                thread_id=threading.get_ident()
            )
            hook_point.execute(ctx)

        # Check that callback was NOT detached
        stats = hook_point.get_callback_statistics()
        assert len(stats) == 1
        assert stats[0]['enabled'] is True
        assert stats[0]['detachment_reason'] is None

    def test_consecutive_violations_reset(self):
        """Test that consecutive violations reset on success."""
        clock = DeterministicClock(start_time=1000.0)

        hook_point = HookPoint(
            name="test_hook",
            timing=HookTiming.BEFORE,
            overhead_budget_ns=1_000_000,
            violation_threshold=3,
            clock=clock
        )

        violation_count = 0

        def intermittent_callback(ctx):
            nonlocal violation_count
            # Alternate between slow and fast
            if violation_count < 2:
                clock.advance(0.002)  # Slow
                violation_count += 1
            else:
                clock.advance(0.0005)  # Fast (should reset)
                violation_count = 0
            return "intermittent"

        hook_point.register(intermittent_callback, priority=10)

        # Execute 10 times (2 slow, then 1 fast, repeat)
        for i in range(10):
            ctx = HookContext(
                timestamp=clock.now(),
                cycle_count=i,
                thread_id=threading.get_ident()
            )
            hook_point.execute(ctx)

        # Should NOT be detached because violations reset
        stats = hook_point.get_callback_statistics()
        assert len(stats) == 1
        assert stats[0]['enabled'] is True


class TestResourceGovernorIntegration:
    """Test ResourceGovernor integration with ToolRegistry."""

    def test_resource_governor_enabled(self):
        """Test that ResourceGovernor is created when enabled."""
        clock = DeterministicClock(start_time=1000.0)
        registry = ToolRegistry(enable_resource_governor=True, clock=clock)

        assert registry._resource_governor is not None
        assert registry._resource_governor._monitoring is True

    def test_resource_governor_disabled(self):
        """Test that ResourceGovernor is not created when disabled."""
        clock = DeterministicClock(start_time=1000.0)
        registry = ToolRegistry(enable_resource_governor=False, clock=clock)

        assert registry._resource_governor is None

    def test_tool_registration_with_governor(self):
        """Test that tools are registered with ResourceGovernor."""
        clock = DeterministicClock(start_time=1000.0)
        registry = ToolRegistry(enable_resource_governor=True, clock=clock)

        config = ToolConfig(tool_name="test")
        tool = PatternPrevalenceQuantifier(config, clock=clock)
        handle = registry.register_tool(tool)

        # Check that tool is registered with governor
        stats = registry._resource_governor.get_statistics()
        assert stats['active_tools'] == 1

    def test_tool_unregistration_with_governor(self):
        """Test that tools are unregistered from ResourceGovernor."""
        clock = DeterministicClock(start_time=1000.0)
        registry = ToolRegistry(enable_resource_governor=True, clock=clock)

        config = ToolConfig(tool_name="test")
        tool = PatternPrevalenceQuantifier(config, clock=clock)
        handle = registry.register_tool(tool)

        # Unregister tool
        registry.unregister_tool(handle.tool_id)

        # Check that tool is unregistered from governor
        stats = registry._resource_governor.get_statistics()
        assert stats['active_tools'] == 0

    def test_resource_governor_statistics(self):
        """Test ResourceGovernor statistics collection."""
        clock = DeterministicClock(start_time=1000.0)
        registry = ToolRegistry(enable_resource_governor=True, clock=clock)

        config = ToolConfig(tool_name="test")
        tool = PatternPrevalenceQuantifier(config, clock=clock)
        handle = registry.register_tool(tool)

        stats = registry._resource_governor.get_statistics()
        assert 'active_tools' in stats
        assert 'total_violations' in stats
        assert 'monitoring_active' in stats
        assert stats['monitoring_active'] is True


class TestLayeredToolMixin:
    """Test LayeredTool mixin validation."""

    def test_pattern_prevalence_quantifier_layer_phase(self):
        """Test PatternPrevalenceQuantifier has correct layer and phase."""
        config = ToolConfig(tool_name="test")
        tool = PatternPrevalenceQuantifier(config)

        assert tool.get_layer() == 2
        assert tool.get_phase() == 1

        meta = tool.get_architectural_metadata()
        assert meta['layer'] == 2
        assert meta['layer_name'] == "Analysis & Intelligence"
        assert meta['phase'] == 1
        assert meta['phase_name'] == "MVP"

    def test_local_entropy_microscope_layer_phase(self):
        """Test LocalEntropyMicroscope has correct layer and phase."""
        config = ToolConfig(tool_name="test")
        tool = LocalEntropyMicroscope(config)

        assert tool.get_layer() == 2
        assert tool.get_phase() == 1

        meta = tool.get_architectural_metadata()
        assert meta['layer'] == 2
        assert meta['layer_name'] == "Analysis & Intelligence"
        assert meta['phase'] == 1
        assert meta['phase_name'] == "MVP"

    def test_event_signature_classifier_layer_phase(self):
        """Test EventSignatureClassifier has correct layer and phase."""
        config = ToolConfig(tool_name="test")
        tool = EventSignatureClassifier(config)

        assert tool.get_layer() == 1
        assert tool.get_phase() == 1

        meta = tool.get_architectural_metadata()
        assert meta['layer'] == 1
        assert meta['layer_name'] == "Sensing & Instrumentation"
        assert meta['phase'] == 1
        assert meta['phase_name'] == "MVP"

    def test_layered_tool_validation(self):
        """Test LayeredTool validation."""
        config = ToolConfig(tool_name="test")
        tool = PatternPrevalenceQuantifier(config)

        # Should not raise
        assert tool.validate_layer_phase() is True


class TestIntegration:
    """Integration tests for all enhancements working together."""

    def test_full_system_integration(self):
        """Test all enhancements working together."""
        # Use deterministic clock for reproducible behavior
        clock = DeterministicClock(start_time=1000.0)

        # Create registry with resource governor
        registry = ToolRegistry(enable_resource_governor=True, clock=clock)

        # Register all MVP tools with clock injection (unique names)
        ppq_config = ToolConfig(tool_name="pattern_quantifier")
        lem_config = ToolConfig(tool_name="entropy_microscope")
        esc_config = ToolConfig(tool_name="event_classifier")

        ppq = PatternPrevalenceQuantifier(ppq_config, clock=clock)
        lem = LocalEntropyMicroscope(lem_config, clock=clock)
        esc = EventSignatureClassifier(esc_config, clock=clock)

        handle1 = registry.register_tool(ppq)
        handle2 = registry.register_tool(lem)
        handle3 = registry.register_tool(esc)

        # Attach all tools
        registry.lifecycle_manager.attach_tool(handle1.tool_id)
        registry.lifecycle_manager.attach_tool(handle2.tool_id)
        registry.lifecycle_manager.attach_tool(handle3.tool_id)

        # All tools should be registered
        assert len(registry._tools) == 3

        # Resource governor should track all tools
        stats = registry._resource_governor.get_statistics()
        assert stats['active_tools'] == 3

        # Clock should be used by all tools
        assert ppq._clock == clock
        assert lem._clock == clock
        assert esc._clock == clock

        # LayeredTool validation should work
        assert ppq.validate_layer_phase() is True
        assert lem.validate_layer_phase() is True
        assert esc.validate_layer_phase() is True

    def test_hook_enforcement_with_registry(self):
        """Test hook enforcement within ToolRegistry context."""
        clock = DeterministicClock(start_time=1000.0)

        # Create a hook with enforcement
        hook_point = HookPoint(
            name="test_hook",
            timing=HookTiming.BEFORE,
            overhead_budget_ns=1_000_000,
            violation_threshold=3,
            clock=clock
        )

        # Add slow callback
        def slow_callback(ctx):
            clock.advance(0.002)
            return "slow"

        hook_point.register(slow_callback, priority=10)

        # Execute multiple times to trigger detachment
        for i in range(5):
            ctx = HookContext(
                timestamp=clock.now(),
                cycle_count=i,
                thread_id=threading.get_ident()
            )
            hook_point.execute(ctx)

        # Verify detachment
        stats = hook_point.get_callback_statistics()
        detached = [s for s in stats if not s['enabled']]
        assert len(detached) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
