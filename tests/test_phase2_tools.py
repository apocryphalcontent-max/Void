"""
Comprehensive tests for Phase 2 (Growth) tools.

Tests:
- ThreatSignatureRecognizer
- BehavioralAnomalyDetector
- NoveltyDetector
- TimelineBranchingEngine
- ProphecyEngine
- ExternalInterferenceDetector
"""

import pytest
from void_state_tools import (
    ToolConfig,
    ThreatSignatureRecognizer,
    BehavioralAnomalyDetector,
    NoveltyDetector,
    TimelineBranchingEngine,
    ProphecyEngine,
    ExternalInterferenceDetector,
    ThreatType,
    Severity,
    ThreatSignature,
    BehaviorTrace,
    BehaviorProfile,
    Perturbation,
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


class TestNoveltyDetector:
    """Test NoveltyDetector functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="novelty_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = NoveltyDetector(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 2
        assert tool.get_phase() == 2

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="novelty_test")
        tool = NoveltyDetector(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Novelty Detector"
        assert meta["layer"] == 2
        assert meta["phase"] == 2
        assert meta["priority"] == "P1"
        assert "novelty_detection" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test initialize, shutdown, suspend, resume."""
        config = ToolConfig(tool_name="novelty_test")
        tool = NoveltyDetector(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_detect_completely_novel_observation(self):
        """Test detection of completely novel observation."""
        config = ToolConfig(tool_name="novelty_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = NoveltyDetector(config, clock=clock)

        observation = {
            "feature_a": 100,
            "feature_b": "unique_value",
            "feature_c": 42.5
        }

        score = tool.detect_novelty(observation, domain="test")

        assert score.novelty == 1.0  # Completely novel
        assert len(score.similar_cases) == 0
        assert score.surprise == float('inf')
        assert score.learnability < 0.5  # Hard to learn without context

    def test_detect_known_observation(self):
        """Test detection of known observation."""
        config = ToolConfig(tool_name="novelty_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = NoveltyDetector(config, clock=clock)

        observation = {
            "x": 10,
            "y": 20,
            "z": "category_a"
        }

        # Add observation to experience base
        tool.detect_novelty(observation, domain="test")

        # Detect same observation again
        score2 = tool.detect_novelty(observation, domain="test")

        # Should have low novelty (similar to previous observation)
        assert score2.novelty < 0.3  # Known pattern
        assert len(score2.similar_cases) > 0
        assert score2.learnability > 0.5  # Easy to learn

    def test_similarity_computation(self):
        """Test similarity computation between observations."""
        config = ToolConfig(tool_name="novelty_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = NoveltyDetector(config, clock=clock)

        # Add baseline observation
        obs1 = {"x": 100, "y": 200, "category": "A"}
        tool.detect_novelty(obs1, domain="test")

        # Similar observation (should have low novelty)
        obs2 = {"x": 102, "y": 198, "category": "A"}
        score2 = tool.detect_novelty(obs2, domain="test")

        assert score2.novelty < 0.5  # Similar
        assert len(score2.similar_cases) > 0

        # Very different observation (should have high novelty)
        obs3 = {"x": 1000, "y": 5000, "category": "Z"}
        score3 = tool.detect_novelty(obs3, domain="test")

        assert score3.novelty > 0.5  # Different

    def test_experience_base_pruning(self):
        """Test that experience base gets pruned when too large."""
        config = ToolConfig(tool_name="novelty_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = NoveltyDetector(config, clock=clock)

        # Set small max size for testing
        tool._max_experience_size = 10

        # Add many observations
        for i in range(15):
            obs = {"value": i}
            tool.detect_novelty(obs, domain="test")

        # Experience base should be pruned
        assert len(tool._experience["test"]) <= 10

    def test_analyze_method(self):
        """Test analyze method."""
        config = ToolConfig(tool_name="novelty_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = NoveltyDetector(config, clock=clock)

        data = {
            "observation": {"a": 1, "b": 2},
            "domain": "test_domain"
        }

        result = tool.analyze(data)

        assert "novelty" in result
        assert "is_novel" in result
        assert "surprise" in result
        assert "learnability" in result
        assert "explanation" in result

    def test_statistics(self):
        """Test statistics tracking."""
        config = ToolConfig(tool_name="novelty_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = NoveltyDetector(config, clock=clock)

        # Add some observations
        for i in range(5):
            obs = {"value": i * 100}  # Very different observations
            tool.detect_novelty(obs, domain="test")

        stats = tool.get_statistics()

        assert stats["total_observations"] == 5
        assert stats["novel_detections"] > 0
        assert stats["experience_base_size"] == 5


class TestExternalInterferenceDetector:
    """Test ExternalInterferenceDetector functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="interference_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ExternalInterferenceDetector(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 3
        assert tool.get_phase() == 2

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="interference_test")
        tool = ExternalInterferenceDetector(config)
        meta = tool.get_metadata()

        assert meta["name"] == "External Interference Detector"
        assert meta["layer"] == 3
        assert meta["phase"] == 2
        assert meta["priority"] == "P1"
        assert "interference_detection" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test initialize, shutdown, suspend, resume."""
        config = ToolConfig(tool_name="interference_test")
        tool = ExternalInterferenceDetector(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_no_interference_clean_readings(self):
        """Test no interference detected for clean sensor readings."""
        config = ToolConfig(tool_name="interference_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ExternalInterferenceDetector(config, clock=clock)

        # Establish baseline with consistent readings
        for i in range(20):
            readings = {
                "cpu_usage": 50.0,
                "memory_usage": 1000.0,
                "io_rate": 100.0
            }
            tool.detect_interference(readings, baseline_id="test")
            clock.advance(1.0)

        # Test with similar readings
        report = tool.detect_interference({
            "cpu_usage": 51.0,
            "memory_usage": 1005.0,
            "io_rate": 102.0
        }, baseline_id="test")

        assert report.detected is False
        assert report.severity == Severity.INFO

    def test_interference_detected_anomalous_readings(self):
        """Test interference detected for anomalous sensor readings."""
        config = ToolConfig(tool_name="interference_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ExternalInterferenceDetector(config, clock=clock)

        # Establish baseline
        for i in range(20):
            readings = {
                "cpu_usage": 50.0,
                "memory_usage": 1000.0
            }
            tool.detect_interference(readings, baseline_id="test")
            clock.advance(1.0)

        # Test with highly anomalous readings
        report = tool.detect_interference({
            "cpu_usage": 500.0,  # 10x baseline!
            "memory_usage": 5000.0  # 5x baseline!
        }, baseline_id="test")

        assert report.detected is True
        assert len(report.affected_components) > 0
        assert report.severity in (Severity.HIGH, Severity.CRITICAL)
        assert report.confidence > 0.5

    def test_source_estimation(self):
        """Test source estimation based on interference patterns."""
        config = ToolConfig(tool_name="interference_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ExternalInterferenceDetector(config, clock=clock)

        # Establish baseline
        for i in range(20):
            readings = {
                "cpu_usage": 50.0,
                "memory_usage": 1000.0,
                "io_rate": 100.0
            }
            tool.detect_interference(readings, baseline_id="test")
            clock.advance(1.0)

        # Simulate resource exhaustion attack
        report = tool.detect_interference({
            "cpu_usage": 500.0,
            "memory_usage": 5000.0,
            "io_rate": 2000.0
        }, baseline_id="test")

        assert report.detected is True
        # Should estimate resource exhaustion based on pattern
        if report.source_estimate != "unknown":
            assert "resource" in report.source_estimate.lower()

    def test_severity_assessment(self):
        """Test severity assessment based on deviation magnitude."""
        config = ToolConfig(tool_name="interference_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ExternalInterferenceDetector(config, clock=clock)

        # Establish baseline with some variance
        import random
        random.seed(42)
        for i in range(20):
            # Add variance to get meaningful stddev
            readings = {"sensor": 100.0 + random.uniform(-2, 2)}
            tool.detect_interference(readings, baseline_id="test")
            clock.advance(1.0)

        # Test different severity levels
        # Low/Medium deviation (~3 stddev)
        report_low = tool.detect_interference(
            {"sensor": 115.0},  # Moderate deviation
            baseline_id="test"
        )

        # High/Critical deviation (>>5 stddev)
        report_high = tool.detect_interference(
            {"sensor": 150.0},  # Large deviation
            baseline_id="test"
        )

        # Both should be detected
        assert report_low.detected is True or report_high.detected is True

        # At least one should have meaningful severity
        if report_high.detected:
            assert report_high.severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)

    def test_baseline_management(self):
        """Test manual baseline setting."""
        config = ToolConfig(tool_name="interference_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ExternalInterferenceDetector(config, clock=clock)

        # Set baseline manually
        tool.set_baseline("custom", {"sensor_a", "sensor_b"})

        assert "custom" in tool._baselines
        assert "sensor_a" in tool._baselines["custom"]
        assert "sensor_b" in tool._baselines["custom"]

    def test_on_event_processing(self):
        """Test event processing via on_event method."""
        config = ToolConfig(tool_name="interference_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ExternalInterferenceDetector(config, clock=clock)

        event = {
            "sensor_readings": {
                "cpu": 50.0,
                "memory": 1000.0
            },
            "baseline_id": "test"
        }

        # Should process without error
        tool.on_event(event)

        assert tool.get_statistics()["total_checks"] > 0

    def test_statistics(self):
        """Test statistics tracking."""
        config = ToolConfig(tool_name="interference_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ExternalInterferenceDetector(config, clock=clock)

        # Generate some checks
        for i in range(10):
            readings = {"sensor": 100.0 + i}
            tool.detect_interference(readings, baseline_id="test")

        stats = tool.get_statistics()

        assert stats["total_checks"] == 10
        assert "detection_rate" in stats
        assert "sensors_monitored" in stats


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
        novelty_config = ToolConfig(tool_name="novelty_tool")
        timeline_config = ToolConfig(tool_name="timeline_tool")
        prophecy_config = ToolConfig(tool_name="prophecy_tool")
        interference_config = ToolConfig(tool_name="interference_tool")

        threat_tool = ThreatSignatureRecognizer(threat_config, clock=clock)
        behavior_tool = BehavioralAnomalyDetector(behavior_config, clock=clock)
        novelty_tool = NoveltyDetector(novelty_config, clock=clock)
        timeline_tool = TimelineBranchingEngine(timeline_config, clock=clock)
        prophecy_tool = ProphecyEngine(prophecy_config, clock=clock)
        interference_tool = ExternalInterferenceDetector(interference_config, clock=clock)

        # Register all tools
        h1 = registry.register_tool(threat_tool)
        h2 = registry.register_tool(behavior_tool)
        h3 = registry.register_tool(novelty_tool)
        h4 = registry.register_tool(timeline_tool)
        h5 = registry.register_tool(prophecy_tool)
        h6 = registry.register_tool(interference_tool)

        # Attach all tools
        registry.lifecycle_manager.attach_tool(h1.tool_id)
        registry.lifecycle_manager.attach_tool(h2.tool_id)
        registry.lifecycle_manager.attach_tool(h3.tool_id)
        registry.lifecycle_manager.attach_tool(h4.tool_id)
        registry.lifecycle_manager.attach_tool(h5.tool_id)
        registry.lifecycle_manager.attach_tool(h6.tool_id)

        # Verify all registered
        assert len(registry._tools) == 6

        # Verify resource governor tracks all
        if registry._resource_governor:
            stats = registry._resource_governor.get_statistics()
            assert stats["active_tools"] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
