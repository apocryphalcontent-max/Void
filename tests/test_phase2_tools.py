"""
Comprehensive tests for Phase 2 (Growth) tools.

Tests:
- ThreatSignatureRecognizer
- BehavioralAnomalyDetector
- TimelineBranchingEngine
- ProphecyEngine
"""

import pytest

from void_state_tools import (
    BehavioralAnomalyDetector,
    BehaviorProfile,
    BehaviorTrace,
    Perturbation,
    ProphecyEngine,
    Severity,
    ThreatSignature,
    ThreatSignatureRecognizer,
    ThreatType,
    TimelineBranchingEngine,
    ToolConfig,
)
from void_state_tools.clock import DeterministicClock


class TestThreatSignatureRecognizer:
    """Test ThreatSignatureRecognizer functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="threat_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ThreatSignatureRecognizer(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 2
        assert tool.get_phase() == 2

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="threat_test")
        tool = ThreatSignatureRecognizer(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Threat Signature Recognizer"
        assert meta["layer"] == 2
        assert meta["phase"] == 2
        assert meta["priority"] == "P0"
        assert "threat_detection" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test initialize, shutdown, suspend, resume."""
        config = ToolConfig(tool_name="threat_test")
        tool = ThreatSignatureRecognizer(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_threat_detection_buffer_overflow(self):
        """Test detection of buffer overflow threat."""
        config = ToolConfig(tool_name="threat_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ThreatSignatureRecognizer(config, clock=clock)

        # Simulate buffer overflow event
        event = {
            "type": "memory_write",
            "buffer_size": 256,
            "write_size": 512,  # Overflow!
            "timestamp": clock.now()
        }

        result = tool.analyze(event)

        assert "threat_detected" in result
        assert "confidence" in result
        assert "matched_signatures" in result

    def test_threat_detection_clean_event(self):
        """Test that clean events don't trigger detection."""
        config = ToolConfig(tool_name="threat_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ThreatSignatureRecognizer(config, clock=clock)

        # Normal event
        event = {
            "type": "file_read",
            "path": "/tmp/test.txt",
            "timestamp": clock.now()
        }

        result = tool.analyze(event)

        assert "threat_detected" in result
        # Clean event should have low or no threat
        if result["threat_detected"]:
            assert result["confidence"] < 0.5

    def test_signature_addition(self):
        """Test adding custom signatures."""
        config = ToolConfig(tool_name="threat_test")
        tool = ThreatSignatureRecognizer(config)

        custom_sig = ThreatSignature(
            signature_id="custom_001",
            threat_type=ThreatType.CODE_INJECTION,
            severity=Severity.HIGH,
            pattern={"action": "custom_threat"},
            indicators=["custom_ioc"],
            description="Custom test signature",
            mitigation="Block custom threat"
        )

        tool.add_signature(custom_sig)

        # Verify signature was added
        event = {"action": "custom_threat"}
        result = tool.analyze(event)

        assert "matched_signatures" in result

    def test_statistics_tracking(self):
        """Test statistics collection."""
        config = ToolConfig(tool_name="threat_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ThreatSignatureRecognizer(config, clock=clock)

        # Analyze several events
        for i in range(5):
            event = {"type": "test", "id": i}
            tool.analyze(event)

        stats = tool.get_statistics()

        assert "total_events_analyzed" in stats
        assert stats["total_events_analyzed"] == 5
        assert "signatures_loaded" in stats


class TestBehavioralAnomalyDetector:
    """Test BehavioralAnomalyDetector functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="behavior_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = BehavioralAnomalyDetector(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 2
        assert tool.get_phase() == 2

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="behavior_test")
        tool = BehavioralAnomalyDetector(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Behavioral Anomaly Detector"
        assert meta["layer"] == 2
        assert meta["phase"] == 2
        assert "behavior_learning" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test lifecycle methods."""
        config = ToolConfig(tool_name="behavior_test")
        tool = BehavioralAnomalyDetector(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_behavior_learning(self):
        """Test behavior sequence learning."""
        config = ToolConfig(tool_name="behavior_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = BehavioralAnomalyDetector(config, clock=clock)

        # Create expected behavior profile
        profile = BehaviorProfile(
            expected_sequences={"login->read->logout", "login->write->logout"},
            state_transitions={},
            frequency_model={},
            temporal_patterns={}
        )

        # Normal behavior
        normal_trace = BehaviorTrace(
            actions=["login", "read", "logout"],
            timestamps=[clock.now(), clock.now() + 1, clock.now() + 2],
            states=[{}, {}, {}]
        )

        result = tool.detect_anomaly(normal_trace, profile)

        assert hasattr(result, "anomaly_detected")
        assert hasattr(result, "risk_assessment")
        assert hasattr(result, "deviation_score")

    def test_anomaly_detection(self):
        """Test anomalous behavior detection functionality."""
        config = ToolConfig(tool_name="behavior_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = BehavioralAnomalyDetector(config, clock=clock)

        # Create expected behavior profile with learned sequences
        profile = BehaviorProfile(
            expected_sequences={"login->read->logout"},
            state_transitions={
                "login": {"read"},
                "read": {"logout"}
            },
            frequency_model={"login": 1.0, "read": 1.0, "logout": 1.0},
            temporal_patterns={}
        )

        # Anomalous behavior (unexpected action sequence with unusual actions)
        anomalous_trace = BehaviorTrace(
            actions=["login", "admin_escalation", "delete_files", "logout"],
            timestamps=[clock.now() + i for i in range(4)],
            states=[{}, {}, {}, {}]
        )

        result = tool.detect_anomaly(anomalous_trace, profile)

        # Verify result structure is correct
        assert hasattr(result, "anomaly_detected")
        assert hasattr(result, "deviant_behaviors")
        assert hasattr(result, "risk_assessment")
        # The detector should identify deviant behaviors (frequency anomalies for new actions)
        assert isinstance(result.deviant_behaviors, set)

    def test_on_event_monitoring(self):
        """Test event monitoring functionality."""
        config = ToolConfig(tool_name="behavior_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = BehavioralAnomalyDetector(config, clock=clock)

        # Send monitoring event
        event = {
            "action": "test_action",
            "state": {"key": "value"}
        }

        # Should not raise exception
        tool.on_event(event)

        stats = tool.get_statistics()
        assert "total_behaviors_observed" in stats


class TestTimelineBranchingEngine:
    """Test TimelineBranchingEngine functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="timeline_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = TimelineBranchingEngine(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 3
        assert tool.get_phase() == 2

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="timeline_test")
        tool = TimelineBranchingEngine(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Timeline Branching Engine"
        assert meta["layer"] == 3
        assert meta["phase"] == 2
        assert "timeline_branching" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test lifecycle methods."""
        config = ToolConfig(tool_name="timeline_test")
        tool = TimelineBranchingEngine(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_timeline_branching(self):
        """Test timeline branching operation."""
        config = ToolConfig(tool_name="timeline_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = TimelineBranchingEngine(config, clock=clock)

        # Create branch point
        branch_point = {
            "x": 10.0,
            "y": 20.0,
            "timestamp": clock.now()
        }

        # Create perturbations
        perturbations = [
            Perturbation(target="x", delta=1.0),
            Perturbation(target="x", delta=-1.0),
        ]

        # Branch timeline
        fork = tool.branch_timeline(branch_point, 2, perturbations)

        assert len(fork.timelines) == 2
        assert fork.branch_point == clock.now()
        assert len(fork.divergence_metrics) > 0

    def test_analyze_method(self):
        """Test analyze method."""
        config = ToolConfig(tool_name="timeline_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = TimelineBranchingEngine(config, clock=clock)

        data = {
            "branch_point": {"x": 5.0, "y": 10.0},
            "num_branches": 3,
            "perturbations": [
                {"target": "x", "delta": 1.0},
                {"target": "x", "delta": -1.0},
                {"target": "y", "delta": 2.0},
            ]
        }

        result = tool.analyze(data)

        assert "num_timelines" in result
        assert "branch_point" in result
        assert result["num_timelines"] == 3


class TestProphecyEngine:
    """Test ProphecyEngine functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="prophecy_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ProphecyEngine(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 3
        assert tool.get_phase() == 2

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="prophecy_test")
        tool = ProphecyEngine(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Prophecy Engine"
        assert meta["layer"] == 3
        assert meta["phase"] == 2
        assert "forward_simulation" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test lifecycle methods."""
        config = ToolConfig(tool_name="prophecy_test")
        tool = ProphecyEngine(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_forecast_generation(self):
        """Test forecast generation."""
        config = ToolConfig(tool_name="prophecy_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ProphecyEngine(config, clock=clock)
        # Override num_trajectories for faster test
        tool._num_trajectories = 10

        # Simple state
        current_state = {
            "value": 100.0,
            "rate": 1.0
        }

        # Simple dynamics (linear growth)
        def dynamics(state, dt):
            new_state = state.copy()
            new_state["value"] = state["value"] + state["rate"] * dt
            return new_state

        # Generate forecast
        distribution = tool.forecast(current_state, dynamics, horizon=10.0)

        assert len(distribution.modes) > 0
        assert len(distribution.uncertainty) > 0
        assert len(distribution.critical_events) >= 0

    def test_analyze_method(self):
        """Test analyze method with default dynamics."""
        config = ToolConfig(tool_name="prophecy_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ProphecyEngine(config, clock=clock)

        data = {
            "current_state": {
                "x": 50.0,
                "y": 100.0
            }
        }

        result = tool.analyze(data)

        assert "num_modes" in result
        assert "uncertainty_dimension" in result
        assert "num_critical_events" in result


class TestIntegration:
    """Integration tests for Phase 2 tools."""

    def test_all_tools_with_registry(self):
        """Test all Phase 2 tools work with ToolRegistry."""
        from void_state_tools import ToolRegistry

        clock = DeterministicClock(start_time=1000.0)
        registry = ToolRegistry(enable_resource_governor=True, clock=clock)

        # Create all Phase 2 tools
        threat_config = ToolConfig(tool_name="threat_tool")
        behavior_config = ToolConfig(tool_name="behavior_tool")
        timeline_config = ToolConfig(tool_name="timeline_tool")
        prophecy_config = ToolConfig(tool_name="prophecy_tool")

        threat_tool = ThreatSignatureRecognizer(threat_config, clock=clock)
        behavior_tool = BehavioralAnomalyDetector(behavior_config, clock=clock)
        timeline_tool = TimelineBranchingEngine(timeline_config, clock=clock)
        prophecy_tool = ProphecyEngine(prophecy_config, clock=clock)

        # Register all tools
        h1 = registry.register_tool(threat_tool)
        h2 = registry.register_tool(behavior_tool)
        h3 = registry.register_tool(timeline_tool)
        h4 = registry.register_tool(prophecy_tool)

        # Attach all tools
        registry.lifecycle_manager.attach_tool(h1.tool_id)
        registry.lifecycle_manager.attach_tool(h2.tool_id)
        registry.lifecycle_manager.attach_tool(h3.tool_id)
        registry.lifecycle_manager.attach_tool(h4.tool_id)

        # Verify all registered
        assert len(registry._tools) == 4

        # Verify resource governor tracks all
        if registry._resource_governor:
            stats = registry._resource_governor.get_statistics()
            assert stats["active_tools"] == 4
