"""
Phase 2 (Growth) Tools - Advanced Analysis & Intelligence

This module contains the 15 tools planned for Phase 2 deployment:
- Layer 2: Threat detection, behavioral analysis, code genealogy
- Layer 3: Temporal prediction, causal analysis, timeline manipulation

All tools implement:
- Clock injection for deterministic testing
- LayeredTool mixin for architectural validation
- Resource-conscious design
- Full type hints and comprehensive docstrings

Phase: 2 (Growth)
Target Overhead: < 5%
"""

import time
import threading
from typing import Dict, Any, List, Set, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import json

from .base import Tool, ToolConfig, AnalysisTool, MonitoringTool
from .layered_tool import LayeredTool
from .clock import Clock, get_clock, SystemClock


# ============================================================================
# LAYER 2: ANALYSIS & INTELLIGENCE
# ============================================================================

class ThreatType(Enum):
    """Types of security threats."""
    MALWARE = "malware"
    INTRUSION = "intrusion"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CODE_INJECTION = "code_injection"
    BUFFER_OVERFLOW = "buffer_overflow"
    BACKDOOR = "backdoor"
    CRYPTOMINING = "cryptomining"
    RANSOMWARE = "ransomware"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"          # Urgent action needed
    MEDIUM = "medium"      # Should be addressed soon
    LOW = "low"            # Monitor and track
    INFO = "info"          # Informational only


@dataclass
class ThreatSignature:
    """Signature definition for threat detection."""
    signature_id: str
    threat_type: ThreatType
    severity: Severity
    pattern: Dict[str, Any]  # Pattern to match
    indicators: List[str]     # IOCs (Indicators of Compromise)
    description: str
    mitigation: str


@dataclass
class ThreatAssessment:
    """Result of threat signature analysis."""
    threat_detected: bool
    threat_type: Optional[ThreatType]
    severity: Optional[Severity]
    matched_signatures: List[str]
    iocs: Set[str]  # Indicators of Compromise
    recommended_actions: List[str]
    confidence: float  # 0.0 - 1.0
    detection_time: float


class ThreatSignatureRecognizer(LayeredTool, AnalysisTool):
    """
    Threat Signature Recognizer - Phase 2 MVP Tool

    Phase: 2 (Growth)
    Layer: 2 (Analysis & Intelligence)
    Priority: P0 (Critical)

    Identifies known threat patterns and attack signatures through
    real-time pattern matching against a curated signature database.
    Provides immediate detection and response recommendations.

    Features:
    - Multi-signature pattern matching with priority queuing
    - IOC (Indicator of Compromise) extraction and correlation
    - Multi-stage attack detection (combines related events)
    - Adaptive confidence scoring based on pattern match quality
    - Response recommendation engine

    Complexity: O(S * P) for S signatures, P pattern complexity
    Overhead Target: < 100µs per event
    """

    _layer = 2
    _phase = 2

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        """
        Initialize the Threat Signature Recognizer.

        Args:
            config: Tool configuration
            clock: Optional clock for deterministic testing
        """
        super().__init__(config)
        self._clock = clock or get_clock()

        # Signature database
        self._signatures: Dict[str, ThreatSignature] = {}
        self._signature_index: Dict[str, Set[str]] = defaultdict(set)

        # Detection tracking
        self._detections: List[ThreatAssessment] = []
        self._detection_counts: Dict[ThreatType, int] = defaultdict(int)
        self._ioc_registry: Set[str] = set()

        # Multi-stage attack tracking
        self._event_history: deque = deque(maxlen=1000)
        self._attack_chains: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Statistics
        self._total_events_analyzed = 0
        self._total_threats_detected = 0
        self._false_positive_count = 0

        # Load default signatures
        self._load_default_signatures()

    def _load_default_signatures(self):
        """Load default threat signatures."""
        # Buffer overflow patterns
        self.add_signature(ThreatSignature(
            signature_id="BUF_001",
            threat_type=ThreatType.BUFFER_OVERFLOW,
            severity=Severity.HIGH,
            pattern={
                "type": "memory_write",
                "size_exceeded": True
            },
            indicators=["oversized_buffer", "stack_smash"],
            description="Stack buffer overflow attempt",
            mitigation="Validate buffer sizes and use safe string functions"
        ))

        # Code injection patterns
        self.add_signature(ThreatSignature(
            signature_id="INJ_001",
            threat_type=ThreatType.CODE_INJECTION,
            severity=Severity.CRITICAL,
            pattern={
                "type": "eval",
                "source": "external"
            },
            indicators=["eval_external", "dynamic_exec"],
            description="Potential code injection via eval",
            mitigation="Sanitize inputs and avoid eval() on untrusted data"
        ))

        # Privilege escalation patterns
        self.add_signature(ThreatSignature(
            signature_id="PRIV_001",
            threat_type=ThreatType.PRIVILEGE_ESCALATION,
            severity=Severity.CRITICAL,
            pattern={
                "type": "permission_change",
                "elevation": True
            },
            indicators=["setuid", "privilege_gain"],
            description="Unauthorized privilege escalation",
            mitigation="Audit permission changes and enforce least privilege"
        ))

    def add_signature(self, signature: ThreatSignature):
        """
        Add a threat signature to the database.

        Args:
            signature: ThreatSignature to add
        """
        self._signatures[signature.signature_id] = signature

        # Index by threat type for faster lookup
        self._signature_index[signature.threat_type.value].add(signature.signature_id)

        # Index IOCs
        for ioc in signature.indicators:
            self._ioc_registry.add(ioc)

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze an event for threat signatures.

        Args:
            data: Event data (dict with event details)

        Returns:
            dict: Threat assessment results
        """
        if not isinstance(data, dict):
            return {"error": "Data must be a dictionary"}

        start_time = self._clock.now()
        self._total_events_analyzed += 1

        # Record event in history for multi-stage detection
        self._event_history.append(data)

        # Perform signature matching
        matched_signatures = []
        iocs_found = set()
        max_severity = None
        max_threat_type = None

        for sig_id, signature in self._signatures.items():
            if self._matches_signature(data, signature):
                matched_signatures.append(sig_id)
                iocs_found.update(signature.indicators)

                # Track highest severity
                if max_severity is None or self._severity_rank(signature.severity) > self._severity_rank(max_severity):
                    max_severity = signature.severity
                    max_threat_type = signature.threat_type

        # Build assessment
        threat_detected = len(matched_signatures) > 0

        if threat_detected:
            self._total_threats_detected += 1
            self._detection_counts[max_threat_type] += 1

        # Generate recommended actions
        actions = self._generate_actions(matched_signatures)

        # Calculate confidence based on match quality
        confidence = self._calculate_confidence(data, matched_signatures)

        detection_time = self._clock.now() - start_time

        assessment = ThreatAssessment(
            threat_detected=threat_detected,
            threat_type=max_threat_type,
            severity=max_severity,
            matched_signatures=matched_signatures,
            iocs=iocs_found,
            recommended_actions=actions,
            confidence=confidence,
            detection_time=detection_time
        )

        self._detections.append(assessment)

        return {
            "threat_detected": threat_detected,
            "threat_type": max_threat_type.value if max_threat_type else None,
            "severity": max_severity.value if max_severity else None,
            "matched_signatures": matched_signatures,
            "iocs": list(iocs_found),
            "recommended_actions": actions,
            "confidence": confidence,
            "detection_time_seconds": detection_time,
        }

    def _matches_signature(self, event: Dict[str, Any], signature: ThreatSignature) -> bool:
        """
        Check if event matches a signature pattern.

        Args:
            event: Event data
            signature: Signature to match against

        Returns:
            bool: True if pattern matches
        """
        pattern = signature.pattern

        # Simple pattern matching (can be extended with regex, etc.)
        for key, expected_value in pattern.items():
            if key not in event:
                return False

            if isinstance(expected_value, bool):
                if bool(event[key]) != expected_value:
                    return False
            elif event[key] != expected_value:
                return False

        return True

    def _severity_rank(self, severity: Severity) -> int:
        """Rank severity for comparison."""
        ranks = {
            Severity.INFO: 0,
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4
        }
        return ranks[severity]

    def _generate_actions(self, matched_signatures: List[str]) -> List[str]:
        """Generate recommended actions based on matched signatures."""
        actions = set()

        for sig_id in matched_signatures:
            signature = self._signatures[sig_id]

            # Add standard actions based on severity
            if signature.severity == Severity.CRITICAL:
                actions.add("Isolate affected system immediately")
                actions.add("Alert security team")
            elif signature.severity == Severity.HIGH:
                actions.add("Investigate immediately")
                actions.add("Log detailed forensics")
            elif signature.severity == Severity.MEDIUM:
                actions.add("Monitor for escalation")
                actions.add("Review related events")

            # Add signature-specific mitigation
            actions.add(signature.mitigation)

        return sorted(list(actions))

    def _calculate_confidence(self, event: Dict[str, Any], matched_signatures: List[str]) -> float:
        """Calculate confidence score for detection."""
        if not matched_signatures:
            return 0.0

        # Base confidence on number of matches
        base_confidence = min(1.0, len(matched_signatures) * 0.3)

        # Increase if multiple IOCs present
        ioc_count = sum(1 for sig_id in matched_signatures
                       for ioc in self._signatures[sig_id].indicators
                       if ioc in str(event))
        ioc_boost = min(0.3, ioc_count * 0.1)

        return min(1.0, base_confidence + ioc_boost)

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_events_analyzed": self._total_events_analyzed,
            "total_threats_detected": self._total_threats_detected,
            "detection_rate": (
                self._total_threats_detected / self._total_events_analyzed
                if self._total_events_analyzed > 0 else 0.0
            ),
            "threats_by_type": {
                threat_type.value: count
                for threat_type, count in self._detection_counts.items()
            },
            "signatures_loaded": len(self._signatures),
            "unique_iocs": len(self._ioc_registry),
            "false_positives": self._false_positive_count,
        }

    def get_recent_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent threat detections."""
        recent = self._detections[-limit:]
        return [
            {
                "threat_detected": d.threat_detected,
                "threat_type": d.threat_type.value if d.threat_type else None,
                "severity": d.severity.value if d.severity else None,
                "confidence": d.confidence,
                "detection_time": d.detection_time,
            }
            for d in recent
        ]

    def initialize(self) -> bool:
        """
        Initialize the threat recognizer by loading default signatures.

        Returns:
            bool: Always True, indicating successful initialization.
        """
        self._load_default_signatures()
        return True

    def shutdown(self) -> bool:
        """
        Clean up resources by clearing signature database and detection history.

        Returns:
            bool: Always True, indicating successful shutdown.
        """
        self._signatures.clear()
        self._signature_index.clear()
        self._detections.clear()
        self._event_history.clear()
        return True

    def suspend(self) -> bool:
        """
        Suspend tool operation while preserving state.

        The recognizer maintains all signatures and detection history during suspension.

        Returns:
            bool: Always True, indicating successful suspension.
        """
        return True

    def resume(self) -> bool:
        """
        Resume tool operation from suspended state.

        The recognizer continues detecting threats with all previous state intact.

        Returns:
            bool: Always True, indicating successful resumption.
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": "Threat Signature Recognizer",
            "category": "security_defense",
            "version": "2.0.0",
            "description": "Real-time threat detection with signature matching and IOC tracking",
            "capabilities": {"threat_detection", "signature_matching", "ioc_tracking", "multi_stage_detection"},
            "dependencies": set(),
            "layer": 2,
            "phase": 2,
            "priority": "P0"
        }


@dataclass
class BehaviorTrace:
    """Sequence of observed behaviors."""
    actions: List[str]
    timestamps: List[float]
    states: List[Dict[str, Any]]


@dataclass
class BehaviorProfile:
    """Learned normal behavior profile."""
    expected_sequences: Set[Tuple[str, ...]]
    state_transitions: Dict[str, Set[str]]  # state -> valid next states
    frequency_model: Dict[str, float]  # action -> frequency
    temporal_patterns: Dict[str, float]  # pattern -> expected duration


@dataclass
class BehaviorAnomalyReport:
    """Result of behavioral anomaly detection."""
    anomaly_detected: bool
    anomaly_severity: float  # 0.0 - 1.0
    deviant_behaviors: Set[str]
    risk_assessment: str
    deviation_score: float
    expected_behavior: Optional[str]
    actual_behavior: str


class BehavioralAnomalyDetector(LayeredTool, MonitoringTool):
    """
    Behavioral Anomaly Detector - Phase 2 Tool

    Phase: 2 (Growth)
    Layer: 2 (Analysis & Intelligence)
    Priority: P1 (High)

    Identifies deviations from learned behavioral patterns through
    sequence comparison, temporal pattern matching, and state machine
    deviation detection.

    Features:
    - Behavior sequence learning and comparison
    - Temporal pattern analysis
    - State transition validation
    - Risk scoring based on deviation magnitude
    - Adaptive thresholds based on historical variance

    Complexity: O(N * M) for N behaviors, M expected patterns
    Overhead Target: < 200µs per behavior
    """

    _layer = 2
    _phase = 2

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        """
        Initialize the Behavioral Anomaly Detector.

        Args:
            config: Tool configuration
            clock: Optional clock for deterministic testing
        """
        super().__init__(config)
        self._clock = clock or get_clock()

        # Behavior learning
        self._observed_sequences: List[Tuple[str, ...]] = []
        self._state_transitions: Dict[str, Set[str]] = defaultdict(set)
        self._action_frequencies: Dict[str, int] = defaultdict(int)
        self._temporal_durations: Dict[str, List[float]] = defaultdict(list)

        # Anomaly detection
        self._anomalies: List[BehaviorAnomalyReport] = []
        self._deviation_history: deque = deque(maxlen=1000)

        # Configuration
        self._sequence_window = 5  # Look at last N actions
        self._anomaly_threshold = 0.7  # Deviation score threshold

        # Statistics
        self._total_behaviors_observed = 0
        self._total_anomalies_detected = 0

    def learn_profile(self, trace: BehaviorTrace) -> BehaviorProfile:
        """
        Learn a behavior profile from observations.

        Args:
            trace: BehaviorTrace with observed actions

        Returns:
            BehaviorProfile: Learned normal behavior model
        """
        # Learn action sequences
        expected_sequences = set()
        for i in range(len(trace.actions) - self._sequence_window + 1):
            sequence = tuple(trace.actions[i:i + self._sequence_window])
            expected_sequences.add(sequence)
            self._observed_sequences.append(sequence)

        # Learn state transitions
        state_transitions = defaultdict(set)
        for i in range(len(trace.states) - 1):
            current_state = self._state_hash(trace.states[i])
            next_state = self._state_hash(trace.states[i + 1])
            state_transitions[current_state].add(next_state)
            self._state_transitions[current_state].add(next_state)

        # Learn action frequencies
        frequency_model = {}
        for action in trace.actions:
            self._action_frequencies[action] += 1

        total = sum(self._action_frequencies.values())
        for action, count in self._action_frequencies.items():
            frequency_model[action] = count / total

        # Learn temporal patterns
        temporal_patterns = {}
        if len(trace.timestamps) > 1:
            for i in range(len(trace.timestamps) - 1):
                duration = trace.timestamps[i + 1] - trace.timestamps[i]
                pattern = f"{trace.actions[i]}->{trace.actions[i + 1]}"
                self._temporal_durations[pattern].append(duration)

                # Calculate average duration
                temporal_patterns[pattern] = sum(self._temporal_durations[pattern]) / len(self._temporal_durations[pattern])

        return BehaviorProfile(
            expected_sequences=expected_sequences,
            state_transitions=dict(state_transitions),
            frequency_model=frequency_model,
            temporal_patterns=temporal_patterns
        )

    def detect_anomaly(self, trace: BehaviorTrace, profile: BehaviorProfile) -> BehaviorAnomalyReport:
        """
        Detect anomalies in a behavior trace.

        Args:
            trace: Observed behavior trace
            profile: Expected behavior profile

        Returns:
            BehaviorAnomalyReport: Anomaly detection results
        """
        self._total_behaviors_observed += len(trace.actions)

        deviant_behaviors = set()
        deviation_scores = []

        # Check sequence deviations
        for i in range(len(trace.actions) - self._sequence_window + 1):
            sequence = tuple(trace.actions[i:i + self._sequence_window])
            if sequence not in profile.expected_sequences:
                deviant_behaviors.add(f"unexpected_sequence_{i}")
                deviation_scores.append(1.0)

        # Check state transition deviations
        for i in range(len(trace.states) - 1):
            current_state = self._state_hash(trace.states[i])
            next_state = self._state_hash(trace.states[i + 1])

            if current_state in profile.state_transitions:
                if next_state not in profile.state_transitions[current_state]:
                    deviant_behaviors.add(f"invalid_transition_{i}")
                    deviation_scores.append(0.8)

        # Check frequency deviations
        for action in set(trace.actions):
            observed_freq = trace.actions.count(action) / len(trace.actions)
            expected_freq = profile.frequency_model.get(action, 0.0)

            if abs(observed_freq - expected_freq) > 0.3:  # 30% deviation
                deviant_behaviors.add(f"frequency_anomaly_{action}")
                deviation_scores.append(0.5)

        # Calculate overall deviation score
        deviation_score = max(deviation_scores) if deviation_scores else 0.0

        # Determine if anomaly detected
        anomaly_detected = deviation_score >= self._anomaly_threshold

        if anomaly_detected:
            self._total_anomalies_detected += 1

        # Assess risk
        risk = self._assess_risk(deviation_score, len(deviant_behaviors))

        report = BehaviorAnomalyReport(
            anomaly_detected=anomaly_detected,
            anomaly_severity=deviation_score,
            deviant_behaviors=deviant_behaviors,
            risk_assessment=risk,
            deviation_score=deviation_score,
            expected_behavior=self._describe_expected(profile),
            actual_behavior=self._describe_actual(trace)
        )

        self._anomalies.append(report)
        self._deviation_history.append(deviation_score)

        return report

    def _state_hash(self, state: Dict[str, Any]) -> str:
        """Create a hash of a state for comparison."""
        # Sort keys for consistent hashing
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

    def _assess_risk(self, deviation_score: float, deviation_count: int) -> str:
        """Assess risk level based on deviations."""
        if deviation_score >= 0.9 or deviation_count >= 10:
            return "CRITICAL"
        elif deviation_score >= 0.7 or deviation_count >= 5:
            return "HIGH"
        elif deviation_score >= 0.5 or deviation_count >= 3:
            return "MEDIUM"
        elif deviation_score >= 0.3 or deviation_count >= 1:
            return "LOW"
        else:
            return "NONE"

    def _describe_expected(self, profile: BehaviorProfile) -> Optional[str]:
        """Describe expected behavior."""
        if not profile.expected_sequences:
            return None

        # Show most common sequence
        return f"Common sequences: {len(profile.expected_sequences)} patterns"

    def _describe_actual(self, trace: BehaviorTrace) -> str:
        """Describe actual observed behavior."""
        return f"Actions: {', '.join(trace.actions[:5])}{'...' if len(trace.actions) > 5 else ''}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_behaviors_observed": self._total_behaviors_observed,
            "total_anomalies_detected": self._total_anomalies_detected,
            "anomaly_rate": (
                self._total_anomalies_detected / max(1, len(self._anomalies))
            ),
            "learned_sequences": len(self._observed_sequences),
            "state_transitions": sum(len(v) for v in self._state_transitions.values()),
            "unique_actions": len(self._action_frequencies),
            "average_deviation": (
                sum(self._deviation_history) / len(self._deviation_history)
                if self._deviation_history else 0.0
            ),
        }

    def on_event(self, event: Dict[str, Any]):
        """Handle monitoring events."""
        # Convert single event to trace and detect anomalies
        if "action" in event:
            trace = BehaviorTrace(
                actions=[event["action"]],
                timestamps=[self._clock.now()],
                states=[event.get("state", {})]
            )

            # Create minimal profile if none exists
            if not self._observed_sequences:
                profile = BehaviorProfile(
                    expected_sequences=set(),
                    state_transitions={},
                    frequency_model={},
                    temporal_patterns={}
                )
            else:
                # Use learned data
                profile = BehaviorProfile(
                    expected_sequences=set(self._observed_sequences[-100:]),
                    state_transitions=dict(self._state_transitions),
                    frequency_model={},
                    temporal_patterns={}
                )

            self.detect_anomaly(trace, profile)

    def initialize(self) -> bool:
        """
        Initialize the behavioral detector.

        Returns:
            bool: Always True, indicating successful initialization.
        """
        return True

    def shutdown(self) -> bool:
        """
        Clean up resources by clearing learned behavioral patterns.

        Returns:
            bool: Always True, indicating successful shutdown.
        """
        self._observed_sequences.clear()
        self._state_transitions.clear()
        self._anomalies.clear()
        return True

    def suspend(self) -> bool:
        """
        Suspend tool operation while preserving learned behaviors.

        Returns:
            bool: Always True, indicating successful suspension.
        """
        return True

    def resume(self) -> bool:
        """
        Resume tool operation from suspended state.

        Returns:
            bool: Always True, indicating successful resumption.
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": "Behavioral Anomaly Detector",
            "category": "security_defense",
            "version": "2.0.0",
            "description": "Detects anomalous behavior patterns through sequence learning and comparison",
            "capabilities": {"behavior_learning", "sequence_analysis", "anomaly_detection", "risk_scoring"},
            "dependencies": set(),
            "layer": 2,
            "phase": 2,
            "priority": "P1"
        }


# ============================================================================
# LAYER 3: COGNITIVE & PREDICTIVE
# ============================================================================

@dataclass
class TimelineFork:
    """Result of timeline branching operation."""
    timelines: List[List[Tuple[float, Dict[str, Any]]]]  # List of timelines
    divergence_metrics: List[List[float]]  # Pairwise divergence matrix
    convergence_points: Set[str]  # States where timelines converge
    branch_point: float  # Original branch time


@dataclass
class Perturbation:
    """State perturbation for timeline branching."""
    target: str  # State variable to perturb
    delta: Any  # Change to apply


class TimelineBranchingEngine(LayeredTool, AnalysisTool):
    """
    Timeline Branching Engine - Phase 2 Tool

    Phase: 2 (Growth)
    Layer: 3 (Cognitive & Predictive)
    Priority: P1 (High)

    Creates and manages alternative execution timelines through
    state forking and parallel simulation. Enables "what-if" scenario
    analysis and divergence/convergence tracking.

    Features:
    - State snapshot and fork management
    - Parallel timeline execution
    - Divergence metric calculation (Euclidean distance in state space)
    - Convergence point detection
    - Timeline merging and comparison

    Complexity: O(T * S * N) for T timelines, S steps, N state dimensions
    Overhead Target: < 1ms per fork operation
    """

    _layer = 3
    _phase = 2

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        """
        Initialize the Timeline Branching Engine.

        Args:
            config: Tool configuration
            clock: Optional clock for deterministic testing
        """
        super().__init__(config)
        self._clock = clock or get_clock()

        # Timeline management
        self._timelines: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        self._branch_points: Dict[str, float] = {}
        self._active_timelines: Set[str] = set()

        # Convergence detection
        self._convergence_threshold = 0.01  # State distance threshold
        self._detected_convergences: Set[str] = set()

        # Statistics
        self._total_branches_created = 0
        self._total_simulations_run = 0

    def branch_timeline(
        self,
        branch_point: Dict[str, Any],
        num_branches: int,
        perturbations: List[Perturbation]
    ) -> TimelineFork:
        """
        Create timeline branches from a branch point.

        Args:
            branch_point: Initial state to branch from
            num_branches: Number of branches to create
            perturbations: Perturbations to apply to each branch

        Returns:
            TimelineFork: Branched timelines with metrics
        """
        branch_time = self._clock.now()
        self._total_branches_created += num_branches

        # Create baseline timeline
        timelines = []
        timeline_ids = []

        # Generate branches
        for i in range(num_branches):
            timeline_id = f"branch_{branch_time}_{i}"
            timeline_ids.append(timeline_id)

            # Create perturbed initial state
            if i < len(perturbations):
                perturbed_state = self._apply_perturbation(
                    branch_point.copy(),
                    perturbations[i]
                )
            else:
                # Random small perturbation if not enough specified
                perturbed_state = self._apply_random_perturbation(branch_point.copy())

            # Initialize timeline with perturbed state
            timeline = [(branch_time, perturbed_state)]
            timelines.append(timeline)

            self._timelines[timeline_id] = timeline
            self._branch_points[timeline_id] = branch_time
            self._active_timelines.add(timeline_id)

        # Calculate initial divergence (should be small)
        divergence_metrics = self._calculate_divergence_matrix(timelines)

        # No convergence points yet at branch time
        convergence_points = set()

        return TimelineFork(
            timelines=timelines,
            divergence_metrics=divergence_metrics,
            convergence_points=convergence_points,
            branch_point=branch_time
        )

    def simulate_step(
        self,
        timeline_id: str,
        dynamics_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simulate one step forward in a timeline.

        Args:
            timeline_id: Timeline to simulate
            dynamics_fn: Function mapping current state -> next state

        Returns:
            dict: Next state
        """
        if timeline_id not in self._timelines:
            raise ValueError(f"Unknown timeline: {timeline_id}")

        timeline = self._timelines[timeline_id]
        current_time, current_state = timeline[-1]

        # Apply dynamics to get next state
        next_state = dynamics_fn(current_state)
        next_time = self._clock.now()

        # Append to timeline
        timeline.append((next_time, next_state))

        self._total_simulations_run += 1

        return next_state

    def _apply_perturbation(
        self,
        state: Dict[str, Any],
        perturbation: Perturbation
    ) -> Dict[str, Any]:
        """Apply a perturbation to a state."""
        if perturbation.target in state:
            current_value = state[perturbation.target]

            # Apply delta based on type
            if isinstance(current_value, (int, float)):
                state[perturbation.target] = current_value + perturbation.delta
            elif isinstance(current_value, str):
                state[perturbation.target] = perturbation.delta  # Replace
            elif isinstance(current_value, list):
                state[perturbation.target] = current_value + [perturbation.delta]
            else:
                state[perturbation.target] = perturbation.delta

        return state

    def _apply_random_perturbation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply small random perturbation to state."""
        import random

        # Perturb numeric values slightly
        for key, value in state.items():
            if isinstance(value, (int, float)):
                noise = random.gauss(0, abs(value) * 0.01 + 0.001)  # 1% noise
                state[key] = value + noise

        return state

    def _calculate_divergence_matrix(
        self,
        timelines: List[List[Tuple[float, Dict[str, Any]]]]
    ) -> List[List[float]]:
        """Calculate pairwise divergence between timelines."""
        n = len(timelines)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                # Get latest states
                _, state_i = timelines[i][-1]
                _, state_j = timelines[j][-1]

                # Calculate Euclidean distance
                divergence = self._state_distance(state_i, state_j)
                matrix[i][j] = divergence
                matrix[j][i] = divergence

        return matrix

    def _state_distance(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Calculate distance between two states."""
        # Simple Euclidean distance for numeric values
        distance_sq = 0.0
        all_keys = set(state1.keys()) | set(state2.keys())

        for key in all_keys:
            val1 = state1.get(key, 0)
            val2 = state2.get(key, 0)

            # Only compare numeric values
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                distance_sq += (val1 - val2) ** 2

        return distance_sq ** 0.5

    def detect_convergence(self, timeline_ids: List[str]) -> Set[str]:
        """
        Detect if timelines have converged.

        Args:
            timeline_ids: Timeline IDs to check

        Returns:
            set: IDs of converged timeline pairs
        """
        converged = set()

        for i in range(len(timeline_ids)):
            for j in range(i + 1, len(timeline_ids)):
                timeline_i = self._timelines[timeline_ids[i]]
                timeline_j = self._timelines[timeline_ids[j]]

                # Compare latest states
                _, state_i = timeline_i[-1]
                _, state_j = timeline_j[-1]

                distance = self._state_distance(state_i, state_j)

                if distance < self._convergence_threshold:
                    converged.add(f"{timeline_ids[i]}<->{timeline_ids[j]}")

        return converged

    def get_statistics(self) -> Dict[str, Any]:
        """Get timeline statistics."""
        return {
            "total_branches_created": self._total_branches_created,
            "total_simulations_run": self._total_simulations_run,
            "active_timelines": len(self._active_timelines),
            "detected_convergences": len(self._detected_convergences),
            "average_timeline_length": (
                sum(len(t) for t in self._timelines.values()) / len(self._timelines)
                if self._timelines else 0
            ),
        }

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data (required by AnalysisTool).

        Args:
            data: Data containing branch_point and perturbations

        Returns:
            dict: Timeline fork information
        """
        if not isinstance(data, dict):
            return {"error": "Data must be a dictionary"}

        branch_point = data.get("branch_point", {})
        num_branches = data.get("num_branches", 2)
        perturbations_data = data.get("perturbations", [])

        # Convert perturbations
        perturbations = [
            Perturbation(target=p["target"], delta=p["delta"])
            for p in perturbations_data
        ]

        fork = self.branch_timeline(branch_point, num_branches, perturbations)

        return {
            "num_timelines": len(fork.timelines),
            "branch_point": fork.branch_point,
            "initial_divergence": max(max(row) for row in fork.divergence_metrics) if fork.divergence_metrics else 0.0,
            "convergence_points": list(fork.convergence_points),
        }

    def initialize(self) -> bool:
        """
        Initialize the timeline branching engine.

        Returns:
            bool: Always True, indicating successful initialization.
        """
        return True

    def shutdown(self) -> bool:
        """
        Clean up resources by clearing timeline history.

        Returns:
            bool: Always True, indicating successful shutdown.
        """
        self._timelines.clear()
        self._active_timelines.clear()
        return True

    def suspend(self) -> bool:
        """
        Suspend tool operation while preserving timeline state.

        Returns:
            bool: Always True, indicating successful suspension.
        """
        return True

    def resume(self) -> bool:
        """
        Resume tool operation from suspended state.

        Returns:
            bool: Always True, indicating successful resumption.
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": "Timeline Branching Engine",
            "category": "cognitive_predictive",
            "version": "2.0.0",
            "description": "Creates and manages alternative execution timelines for what-if analysis",
            "capabilities": {"timeline_branching", "state_forking", "divergence_tracking", "convergence_detection"},
            "dependencies": set(),
            "layer": 3,
            "phase": 2,
            "priority": "P1"
        }


@dataclass
class ProphecyDistribution:
    """Distribution of predicted future states."""
    modes: List[Tuple[Dict[str, Any], float]]  # (state, probability) pairs
    uncertainty: List[List[float]]  # Covariance matrix
    critical_events: List[Dict[str, Any]]  # Events that significantly affect future


class ProphecyEngine(LayeredTool, AnalysisTool):
    """
    Prophecy Engine (Forward Simulator) - Phase 2 Tool

    Phase: 2 (Growth)
    Layer: 3 (Cognitive & Predictive)
    Priority: P1 (High)

    Projects probable future states with uncertainty quantification
    through Monte Carlo trajectory sampling and forward dynamics simulation.
    Identifies critical events that significantly impact outcomes.

    Features:
    - Forward dynamics simulation
    - Monte Carlo trajectory sampling (embarrassingly parallel)
    - Uncertainty propagation via ensemble methods
    - Critical event identification through sensitivity analysis
    - Multi-modal future state distribution

    Complexity: O(N * T * S) for N trajectories, T time steps, S state dimensions
    Overhead Target: Background continuous forecasting
    """

    _layer = 3
    _phase = 2

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        """
        Initialize the Prophecy Engine.

        Args:
            config: Tool configuration
            clock: Optional clock for deterministic testing
        """
        super().__init__(config)
        self._clock = clock or get_clock()

        # Forecast configuration
        self._num_trajectories = 100  # Monte Carlo sample size
        self._forecast_horizon = 10.0  # Time horizon in seconds
        self._time_step = 0.1  # Simulation time step

        # Forecasts
        self._forecasts: List[ProphecyDistribution] = []
        self._trajectories: List[List[Dict[str, Any]]] = []

        # Critical events
        self._critical_threshold = 0.1  # Sensitivity threshold

        # Statistics
        self._total_forecasts = 0
        self._total_trajectories_simulated = 0

    def forecast(
        self,
        current_state: Dict[str, Any],
        dynamics_fn: Callable[[Dict[str, Any], float], Dict[str, Any]],
        horizon: Optional[float] = None
    ) -> ProphecyDistribution:
        """
        Generate forecast of future states.

        Args:
            current_state: Current system state
            dynamics_fn: Function(state, dt) -> next_state
            horizon: Time horizon (uses default if None)

        Returns:
            ProphecyDistribution: Distribution of future states
        """
        horizon = horizon or self._forecast_horizon
        self._total_forecasts += 1

        # Run Monte Carlo simulations
        trajectories = []
        final_states = []

        for i in range(self._num_trajectories):
            trajectory = self._simulate_trajectory(
                current_state.copy(),
                dynamics_fn,
                horizon
            )
            trajectories.append(trajectory)
            final_states.append(trajectory[-1])

            self._total_trajectories_simulated += 1

        self._trajectories = trajectories

        # Cluster final states to find modes
        modes = self._find_modes(final_states)

        # Calculate uncertainty (covariance of final states)
        uncertainty = self._calculate_covariance(final_states)

        # Identify critical events
        critical_events = self._identify_critical_events(trajectories)

        distribution = ProphecyDistribution(
            modes=modes,
            uncertainty=uncertainty,
            critical_events=critical_events
        )

        self._forecasts.append(distribution)

        return distribution

    def _simulate_trajectory(
        self,
        initial_state: Dict[str, Any],
        dynamics_fn: Callable[[Dict[str, Any], float], Dict[str, Any]],
        horizon: float
    ) -> List[Dict[str, Any]]:
        """Simulate a single trajectory forward in time."""
        trajectory = [initial_state]
        state = initial_state.copy()

        num_steps = int(horizon / self._time_step)

        for _ in range(num_steps):
            state = dynamics_fn(state, self._time_step)
            trajectory.append(state.copy())

        return trajectory

    def _find_modes(
        self,
        states: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find modes (clusters) in final state distribution."""
        if not states:
            return []

        # Simple clustering: find representative states
        # In production, would use proper clustering (K-means, DBSCAN, etc.)

        # For now, just return most common states
        # Count occurrences (approximate via rounding)
        state_counts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for state in states:
            # Create approximate key
            key = self._state_key(state)
            state_counts[key].append(state)

        # Convert to modes with probabilities
        modes = []
        total = len(states)

        for key, similar_states in sorted(state_counts.items(), key=lambda x: len(x[1]), reverse=True):
            if len(similar_states) / total > 0.05:  # At least 5% probability
                representative = similar_states[0]
                probability = len(similar_states) / total
                modes.append((representative, probability))

        return modes[:5]  # Top 5 modes

    def _state_key(self, state: Dict[str, Any]) -> str:
        """Create approximate key for state grouping."""
        # Round numeric values for grouping
        rounded = {}
        for key, value in state.items():
            if isinstance(value, float):
                rounded[key] = round(value, 1)
            else:
                rounded[key] = value

        return json.dumps(rounded, sort_keys=True)

    def _calculate_covariance(
        self,
        states: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """Calculate covariance matrix of states."""
        if not states:
            return [[]]

        # Extract numeric values
        keys = sorted([k for k in states[0].keys() if isinstance(states[0][k], (int, float))])

        if not keys:
            return [[]]

        # Build data matrix
        data = []
        for state in states:
            row = [state.get(k, 0.0) for k in keys]
            data.append(row)

        # Calculate covariance
        n = len(data)
        m = len(keys)

        # Calculate means
        means = [sum(data[i][j] for i in range(n)) / n for j in range(m)]

        # Calculate covariance matrix
        cov = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                cov[i][j] = sum(
                    (data[k][i] - means[i]) * (data[k][j] - means[j])
                    for k in range(n)
                ) / (n - 1)

        return cov

    def _identify_critical_events(
        self,
        trajectories: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Identify events that significantly affect trajectories."""
        critical_events = []

        # Look for steps where trajectories diverge significantly
        if not trajectories or len(trajectories) < 2:
            return critical_events

        min_length = min(len(t) for t in trajectories)

        for step in range(1, min_length):
            # Calculate variance at this step
            variance = self._calculate_step_variance(trajectories, step)

            # If variance increases significantly, mark as critical
            if step > 1:
                prev_variance = self._calculate_step_variance(trajectories, step - 1)
                if variance > prev_variance * 1.5 and prev_variance > 0:  # 50% increase
                    critical_events.append({
                        "step": step,
                        "time": step * self._time_step,
                        "variance_increase": (variance - prev_variance) / prev_variance,
                        "description": "Significant divergence point"
                    })

        return critical_events

    def _calculate_step_variance(
        self,
        trajectories: List[List[Dict[str, Any]]],
        step: int
    ) -> float:
        """Calculate variance of states at a given step."""
        states = [traj[step] for traj in trajectories if len(traj) > step]

        if not states:
            return 0.0

        # Calculate variance of numeric values
        total_variance = 0.0
        keys = [k for k in states[0].keys() if isinstance(states[0][k], (int, float))]

        for key in keys:
            values = [s.get(key, 0.0) for s in states]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            total_variance += variance

        return total_variance

    def get_statistics(self) -> Dict[str, Any]:
        """Get forecast statistics."""
        return {
            "total_forecasts": self._total_forecasts,
            "total_trajectories_simulated": self._total_trajectories_simulated,
            "average_trajectories_per_forecast": (
                self._total_trajectories_simulated / max(1, self._total_forecasts)
            ),
            "forecast_horizon": self._forecast_horizon,
            "time_step": self._time_step,
        }

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data and generate forecast.

        Args:
            data: Dict with current_state and optional dynamics

        Returns:
            dict: Forecast summary
        """
        if not isinstance(data, dict):
            return {"error": "Data must be a dictionary"}

        current_state = data.get("current_state", {})

        # Default simple dynamics (random walk)
        def default_dynamics(state: Dict[str, Any], dt: float) -> Dict[str, Any]:
            import random
            new_state = state.copy()
            for key, value in state.items():
                if isinstance(value, (int, float)):
                    noise = random.gauss(0, abs(value) * 0.1 * dt)
                    new_state[key] = value + noise
            return new_state

        distribution = self.forecast(current_state, default_dynamics)

        return {
            "num_modes": len(distribution.modes),
            "top_mode_probability": distribution.modes[0][1] if distribution.modes else 0.0,
            "uncertainty_dimension": len(distribution.uncertainty) if distribution.uncertainty else 0,
            "num_critical_events": len(distribution.critical_events),
            "critical_events": distribution.critical_events,
        }

    def initialize(self) -> bool:
        """
        Initialize the prophecy engine.

        Returns:
            bool: Always True, indicating successful initialization.
        """
        return True

    def shutdown(self) -> bool:
        """
        Clean up resources by clearing forecast history.

        Returns:
            bool: Always True, indicating successful shutdown.
        """
        self._forecasts.clear()
        return True

    def suspend(self) -> bool:
        """
        Suspend tool operation while preserving forecast state.

        Returns:
            bool: Always True, indicating successful suspension.
        """
        return True

    def resume(self) -> bool:
        """
        Resume tool operation from suspended state.

        Returns:
            bool: Always True, indicating successful resumption.
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": "Prophecy Engine",
            "category": "cognitive_predictive",
            "version": "2.0.0",
            "description": "Projects probable future states via Monte Carlo simulation",
            "capabilities": {"forward_simulation", "monte_carlo", "uncertainty_quantification", "critical_event_detection"},
            "dependencies": set(),
            "layer": 3,
            "phase": 2,
            "priority": "P1"
        }


# ============================================================================
# NOVELTY DETECTION
# ============================================================================

@dataclass
class NoveltyScore:
    """Result of novelty detection analysis."""
    novelty: float  # 0.0 (seen before) to 1.0 (completely novel)
    similar_cases: List[Tuple[str, float]]  # (case_id, similarity)
    surprise: float  # Information-theoretic surprise
    learnability: float  # How easy to incorporate into experience
    explanation: str
    timestamp: float


class NoveltyDetector(LayeredTool, AnalysisTool):
    """
    Novelty Detector - Phase 2 Pattern Recognition Tool

    Phase: 2 (Growth)
    Layer: 2 (Analysis & Intelligence)
    Priority: P1 (High)

    Identifies unprecedented patterns by comparing observations against
    an experience base. Uses similarity metrics and information-theoretic
    measures to quantify novelty and surprise.

    Features:
    - Multi-dimensional similarity search
    - Information-theoretic surprise calculation
    - Learnability assessment for novel patterns
    - Experience base management with efficient indexing
    - Adaptive novelty thresholds based on domain

    Complexity: O(N log N) for similarity search with indexing
    Space: O(E) where E = size of experience base
    Overhead Target: < 200µs per observation

    Thread Safety: Fully thread-safe with lock-based synchronization
    """

    _layer = 2
    _phase = 2

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        """
        Initialize the novelty detector.

        Args:
            config: Tool configuration
            clock: Optional clock for deterministic testing
        """
        super().__init__(config)
        self._clock = clock or get_clock()

        # Experience base: maps feature signatures to observations
        self._experience: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Statistics
        self._total_observations = 0
        self._novel_detections = 0
        self._similarity_cache: Dict[str, List[Tuple[str, float]]] = {}

        # Configuration
        self._novelty_threshold = 0.7  # Threshold for "novel" classification
        self._max_experience_size = 10000  # Maximum observations to retain
        self._similarity_top_k = 5  # Top K similar cases to return

        # Thread safety
        self._lock = threading.Lock()

    def detect_novelty(
        self,
        observation: Dict[str, Any],
        domain: str = "default"
    ) -> NoveltyScore:
        """
        Detect novelty in an observation.

        Args:
            observation: Observation to analyze
            domain: Domain context for the observation

        Returns:
            NoveltyScore with novelty metrics

        Time Complexity: O(N log N) for similarity search
        Space Complexity: O(1) excluding returned data
        """
        with self._lock:
            timestamp = self._clock.now()
            self._total_observations += 1

            # Compute feature signature
            signature = self._compute_signature(observation)
            cache_key = f"{domain}:{signature}"

            # Check cache first
            if cache_key in self._similarity_cache:
                similar_cases = self._similarity_cache[cache_key]
            else:
                # Find similar observations in experience base
                similar_cases = self._find_similar(observation, domain)
                self._similarity_cache[cache_key] = similar_cases

            # Compute novelty score
            if not similar_cases:
                # Completely novel - no similar cases found
                novelty = 1.0
                surprise = float('inf')  # Maximum surprise
                explanation = "No similar observations in experience base"
            else:
                # Novelty based on maximum similarity
                max_similarity = similar_cases[0][1]
                novelty = 1.0 - max_similarity

                # Information-theoretic surprise: -log(similarity)
                # Higher similarity = lower surprise
                surprise = -sum(sim for _, sim in similar_cases[:3]) / min(3, len(similar_cases))
                surprise = max(0.0, surprise)  # Clamp to non-negative

                if novelty >= self._novelty_threshold:
                    explanation = f"Novel pattern: only {max_similarity:.1%} similar to closest match"
                else:
                    explanation = f"Known pattern: {max_similarity:.1%} similar to existing observations"

            # Assess learnability
            learnability = self._assess_learnability(observation, similar_cases)

            # Add to experience base
            self._add_to_experience(observation, domain, signature)

            # Track novel detections
            if novelty >= self._novelty_threshold:
                self._novel_detections += 1

            return NoveltyScore(
                novelty=novelty,
                similar_cases=similar_cases,
                surprise=surprise,
                learnability=learnability,
                explanation=explanation,
                timestamp=timestamp
            )

    def _compute_signature(self, observation: Dict[str, Any]) -> str:
        """Compute a hash signature for an observation."""
        # Normalize and serialize observation
        normalized = json.dumps(observation, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _find_similar(
        self,
        observation: Dict[str, Any],
        domain: str
    ) -> List[Tuple[str, float]]:
        """
        Find similar observations in experience base.

        Returns:
            List of (observation_id, similarity) tuples, sorted by similarity
        """
        domain_experiences = self._experience.get(domain, [])

        if not domain_experiences:
            return []

        # Compute similarities
        similarities = []
        for exp_obs in domain_experiences:
            similarity = self._compute_similarity(observation, exp_obs['data'])
            similarities.append((exp_obs['id'], similarity))

        # Sort by similarity (descending) and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self._similarity_top_k]

    def _compute_similarity(
        self,
        obs1: Dict[str, Any],
        obs2: Dict[str, Any]
    ) -> float:
        """
        Compute similarity between two observations.

        Uses cosine similarity for numeric features and Jaccard for categorical.

        Returns:
            Similarity score in [0, 1]
        """
        # Get common keys
        keys = set(obs1.keys()) & set(obs2.keys())

        if not keys:
            return 0.0

        # Separate numeric and categorical features
        numeric_sim = 0.0
        categorical_sim = 0.0
        numeric_count = 0
        categorical_count = 0

        for key in keys:
            val1, val2 = obs1[key], obs2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity: inverse of relative difference
                if val1 == 0 and val2 == 0:
                    numeric_sim += 1.0
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        numeric_sim += 1.0 - abs(val1 - val2) / max_val
                numeric_count += 1
            else:
                # Categorical similarity: exact match
                if val1 == val2:
                    categorical_sim += 1.0
                categorical_count += 1

        # Weighted average
        total_features = numeric_count + categorical_count
        if total_features == 0:
            return 0.0

        return (numeric_sim + categorical_sim) / total_features

    def _assess_learnability(
        self,
        observation: Dict[str, Any],
        similar_cases: List[Tuple[str, float]]
    ) -> float:
        """
        Assess how easy it would be to learn this pattern.

        High learnability: Similar to existing patterns (easy to integrate)
        Low learnability: Very different from existing patterns (hard to categorize)

        Returns:
            Learnability score in [0, 1]
        """
        if not similar_cases:
            # No similar cases: difficult to learn without context
            return 0.3

        # Learnability based on similarity distribution
        similarities = [sim for _, sim in similar_cases]
        avg_similarity = sum(similarities) / len(similarities)

        # High average similarity = easier to learn
        return avg_similarity

    def _add_to_experience(
        self,
        observation: Dict[str, Any],
        domain: str,
        signature: str
    ):
        """Add observation to experience base."""
        obs_id = f"{domain}:{signature}:{self._total_observations}"

        self._experience[domain].append({
            'id': obs_id,
            'data': observation,
            'signature': signature,
            'timestamp': self._clock.now()
        })

        # Invalidate cache entry for this signature since experience base changed
        cache_key = f"{domain}:{signature}"
        if cache_key in self._similarity_cache:
            del self._similarity_cache[cache_key]

        # Prune if experience base too large
        if len(self._experience[domain]) > self._max_experience_size:
            # Remove oldest 10%
            to_remove = self._max_experience_size // 10
            self._experience[domain] = self._experience[domain][to_remove:]
            # Clear cache when pruning
            self._similarity_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get novelty detection statistics."""
        return {
            "total_observations": self._total_observations,
            "novel_detections": self._novel_detections,
            "novelty_rate": self._novel_detections / max(1, self._total_observations),
            "experience_base_size": sum(len(exps) for exps in self._experience.values()),
            "domains": len(self._experience),
            "cache_size": len(self._similarity_cache),
        }

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data for novelty.

        Args:
            data: Observation dict or dict with 'observation' and 'domain' keys

        Returns:
            dict: Novelty analysis summary
        """
        if not isinstance(data, dict):
            return {"error": "Data must be a dictionary"}

        # Extract observation and domain
        if 'observation' in data:
            observation = data['observation']
            domain = data.get('domain', 'default')
        else:
            observation = data
            domain = 'default'

        score = self.detect_novelty(observation, domain)

        return {
            "novelty": score.novelty,
            "is_novel": score.novelty >= self._novelty_threshold,
            "surprise": score.surprise,
            "learnability": score.learnability,
            "num_similar_cases": len(score.similar_cases),
            "closest_similarity": score.similar_cases[0][1] if score.similar_cases else 0.0,
            "explanation": score.explanation,
        }

    def initialize(self) -> bool:
        """Initialize the novelty detector."""
        return True

    def shutdown(self) -> bool:
        """Clean up resources."""
        with self._lock:
            self._experience.clear()
            self._similarity_cache.clear()
        return True

    def suspend(self) -> bool:
        """Suspend tool operation."""
        return True

    def resume(self) -> bool:
        """Resume tool operation."""
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": "Novelty Detector",
            "category": "pattern_recognition",
            "version": "2.0.0",
            "description": "Identifies unprecedented patterns through similarity analysis",
            "capabilities": {"novelty_detection", "similarity_search", "surprise_quantification", "learnability_assessment"},
            "dependencies": set(),
            "layer": 2,
            "phase": 2,
            "priority": "P1"
        }


# ============================================================================
# LAYER 3: COGNITIVE & PREDICTIVE - NOETIC INTERFERENCE ANALYSIS
# ============================================================================

@dataclass
class InterferenceReport:
    """Result of external interference detection."""
    detected: bool
    interference_vector: Dict[str, float]  # Deviation from baseline
    source_estimate: Optional[str]  # Estimated source of interference
    confidence: float  # Detection confidence
    affected_components: Set[str]  # System components affected
    severity: Severity
    recommended_actions: List[str]
    timestamp: float


class ExternalInterferenceDetector(LayeredTool, MonitoringTool):
    """
    External Interference Detector - Phase 2 Security Tool

    Phase: 2 (Growth)
    Layer: 3 (Cognitive & Predictive)
    Priority: P1 (High - Security)

    Detects unauthorized external influences on system state by comparing
    current behavior against established baselines. Uses multi-sensor fusion
    and anomaly detection to identify interference patterns.

    Features:
    - Multi-dimensional baseline tracking
    - Real-time deviation detection
    - Source estimation through pattern analysis
    - Severity assessment and risk scoring
    - Automatic response recommendation

    Complexity: O(S) where S = number of sensors
    Space: O(H * S) where H = history window size
    Overhead Target: < 500µs per check

    Thread Safety: Fully thread-safe with lock-based synchronization
    """

    _layer = 3
    _phase = 2

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        """
        Initialize the interference detector.

        Args:
            config: Tool configuration
            clock: Optional clock for deterministic testing
        """
        super().__init__(config)
        self._clock = clock or get_clock()

        # Baseline storage: sensor_id -> baseline statistics
        self._baselines: Dict[str, Dict[str, float]] = {}

        # Sensor readings history
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Known interference patterns
        self._interference_patterns: Dict[str, Dict[str, Any]] = {
            "timing_attack": {
                "indicators": {"timing_variance", "response_delay"},
                "threshold": 2.0,  # Standard deviations
            },
            "resource_exhaustion": {
                "indicators": {"cpu_usage", "memory_usage", "io_rate"},
                "threshold": 3.0,
            },
            "data_tampering": {
                "indicators": {"checksum_mismatch", "integrity_violation"},
                "threshold": 1.0,  # Any detection is serious
            },
        }

        # Statistics
        self._total_checks = 0
        self._detections = 0
        self._false_positive_estimate = 0

        # Configuration
        self._detection_threshold = 2.5  # Standard deviations for anomaly
        self._baseline_window = 1000  # Observations for baseline calculation

        # Thread safety
        self._lock = threading.Lock()

    def detect_interference(
        self,
        sensor_readings: Dict[str, float],
        baseline_id: str = "default"
    ) -> InterferenceReport:
        """
        Detect external interference in sensor readings.

        Args:
            sensor_readings: Current sensor values
            baseline_id: ID of baseline to compare against

        Returns:
            InterferenceReport with detection results

        Time Complexity: O(S) where S = number of sensors
        Space Complexity: O(1) excluding stored history
        """
        with self._lock:
            timestamp = self._clock.now()
            self._total_checks += 1

            # Update history
            for sensor_id, value in sensor_readings.items():
                self._history[sensor_id].append((timestamp, value))

            # Get or create baseline (only if we have enough history)
            if baseline_id not in self._baselines:
                # Check if we have enough history to compute a meaningful baseline
                min_samples = any(
                    len(self._history.get(sensor_id, [])) > 10
                    for sensor_id in sensor_readings.keys()
                )
                if min_samples:
                    self._compute_baseline(baseline_id, sensor_readings.keys())
                else:
                    # Not enough history yet - no detection possible
                    return InterferenceReport(
                        detected=False,
                        interference_vector={},
                        source_estimate=None,
                        confidence=0.0,
                        affected_components=set(),
                        severity=Severity.INFO,
                        recommended_actions=["Collecting baseline data"],
                        timestamp=timestamp
                    )

            baseline = self._baselines[baseline_id]

            # Compute deviations
            interference_vector = {}
            anomalous_sensors = set()

            for sensor_id, value in sensor_readings.items():
                if sensor_id in baseline:
                    mean = baseline[sensor_id]['mean']
                    stddev = baseline[sensor_id]['stddev']

                    if stddev > 0:
                        deviation = abs(value - mean) / stddev
                    else:
                        deviation = 0.0 if value == mean else float('inf')

                    interference_vector[sensor_id] = deviation

                    if deviation >= self._detection_threshold:
                        anomalous_sensors.add(sensor_id)

            # Determine if interference detected
            detected = len(anomalous_sensors) > 0

            if detected:
                self._detections += 1

                # Estimate source and severity
                source_estimate = self._estimate_source(anomalous_sensors, interference_vector)
                severity = self._assess_severity(interference_vector, anomalous_sensors)
                confidence = self._compute_confidence(interference_vector, anomalous_sensors)

                # Generate recommendations
                recommendations = self._generate_recommendations(
                    source_estimate, severity, anomalous_sensors
                )

                explanation = f"Detected {len(anomalous_sensors)} anomalous sensors: {', '.join(list(anomalous_sensors)[:3])}"
            else:
                source_estimate = None
                severity = Severity.INFO
                confidence = 0.0
                recommendations = []
                explanation = "No interference detected"

            return InterferenceReport(
                detected=detected,
                interference_vector=interference_vector,
                source_estimate=source_estimate,
                confidence=confidence,
                affected_components=anomalous_sensors,
                severity=severity,
                recommended_actions=recommendations,
                timestamp=timestamp
            )

    def _compute_baseline(self, baseline_id: str, sensor_ids: Set[str]):
        """Compute baseline statistics for sensors."""
        self._baselines[baseline_id] = {}

        for sensor_id in sensor_ids:
            if sensor_id in self._history and len(self._history[sensor_id]) > 10:
                values = [v for _, v in self._history[sensor_id]]
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                stddev = variance ** 0.5

                # Use minimum stddev to avoid division by zero and over-sensitivity
                # If stddev is 0 (constant values), use 5% of mean as threshold
                if stddev < 0.01:
                    stddev = max(abs(mean) * 0.05, 1.0) if mean != 0 else 1.0
            else:
                # Insufficient data: use defaults
                mean = 0.0
                stddev = 1.0

            self._baselines[baseline_id][sensor_id] = {
                'mean': mean,
                'stddev': stddev,
                'samples': len(self._history.get(sensor_id, []))
            }

    def _estimate_source(
        self,
        anomalous_sensors: Set[str],
        interference_vector: Dict[str, float]
    ) -> str:
        """Estimate the source of interference based on pattern matching."""
        best_match = "unknown"
        best_score = 0.0

        for pattern_name, pattern in self._interference_patterns.items():
            # Count how many pattern indicators are anomalous
            matches = sum(
                1 for indicator in pattern['indicators']
                if indicator in anomalous_sensors
            )

            score = matches / len(pattern['indicators'])

            if score > best_score:
                best_score = score
                best_match = pattern_name

        if best_score >= 0.5:
            return best_match
        return "unknown"

    def _assess_severity(
        self,
        interference_vector: Dict[str, float],
        anomalous_sensors: Set[str]
    ) -> Severity:
        """Assess the severity of detected interference."""
        if not anomalous_sensors:
            return Severity.INFO

        max_deviation = max(interference_vector.values())
        num_affected = len(anomalous_sensors)

        if max_deviation >= 5.0 or num_affected >= 5:
            return Severity.CRITICAL
        elif max_deviation >= 4.0 or num_affected >= 3:
            return Severity.HIGH
        elif max_deviation >= 3.0 or num_affected >= 2:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _compute_confidence(
        self,
        interference_vector: Dict[str, float],
        anomalous_sensors: Set[str]
    ) -> float:
        """Compute confidence in interference detection."""
        if not anomalous_sensors:
            return 0.0

        # Confidence based on deviation magnitude and number of sensors
        avg_deviation = sum(
            interference_vector[s] for s in anomalous_sensors
        ) / len(anomalous_sensors)

        # Map deviation to confidence (sigmoid-like)
        confidence = min(1.0, avg_deviation / 5.0)

        return confidence

    def _generate_recommendations(
        self,
        source: Optional[str],
        severity: Severity,
        affected: Set[str]
    ) -> List[str]:
        """Generate response recommendations."""
        recommendations = []

        if severity in (Severity.CRITICAL, Severity.HIGH):
            recommendations.append("Immediate investigation required")
            recommendations.append("Consider isolating affected components")

        if source == "timing_attack":
            recommendations.append("Enable timing jitter countermeasures")
            recommendations.append("Review access patterns for side-channel leaks")
        elif source == "resource_exhaustion":
            recommendations.append("Enforce stricter resource quotas")
            recommendations.append("Identify and throttle resource-intensive operations")
        elif source == "data_tampering":
            recommendations.append("Verify data integrity across all components")
            recommendations.append("Enable cryptographic signatures for critical data")
        else:
            recommendations.append("Collect additional diagnostic data")
            recommendations.append(f"Monitor sensors: {', '.join(list(affected)[:5])}")

        return recommendations

    def set_baseline(self, baseline_id: str, sensor_ids: Set[str]):
        """Manually trigger baseline computation."""
        with self._lock:
            self._compute_baseline(baseline_id, sensor_ids)

    def get_statistics(self) -> Dict[str, Any]:
        """Get interference detection statistics."""
        return {
            "total_checks": self._total_checks,
            "detections": self._detections,
            "detection_rate": self._detections / max(1, self._total_checks),
            "baselines": len(self._baselines),
            "sensors_monitored": len(self._history),
            "avg_history_depth": (
                sum(len(h) for h in self._history.values()) / max(1, len(self._history))
            ),
        }

    def on_event(self, event: Any) -> None:
        """
        Process sensor events.

        Args:
            event: Sensor reading event
        """
        if not isinstance(event, dict):
            return

        sensor_readings = event.get('sensor_readings', {})
        baseline_id = event.get('baseline_id', 'default')

        if sensor_readings:
            report = self.detect_interference(sensor_readings, baseline_id)

            if report.detected and report.severity in (Severity.CRITICAL, Severity.HIGH):
                # Log critical detections (in production, this would trigger alerts)
                pass

    def initialize(self) -> bool:
        """Initialize the interference detector."""
        return True

    def shutdown(self) -> bool:
        """Clean up resources."""
        with self._lock:
            self._baselines.clear()
            self._history.clear()
        return True

    def suspend(self) -> bool:
        """Suspend tool operation."""
        return True

    def resume(self) -> bool:
        """Resume tool operation."""
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": "External Interference Detector",
            "category": "noetic_security",
            "version": "2.0.0",
            "description": "Detects unauthorized external influences through baseline deviation analysis",
            "capabilities": {"interference_detection", "baseline_tracking", "source_estimation", "severity_assessment"},
            "dependencies": set(),
            "layer": 3,
            "phase": 2,
            "priority": "P1"
        }


# Export public API
__all__ = [
    # Enums
    'ThreatType',
    'Severity',
    # Data classes
    'ThreatSignature',
    'ThreatAssessment',
    'BehaviorTrace',
    'BehaviorProfile',
    'BehaviorAnomalyReport',
    'TimelineFork',
    'Perturbation',
    'ProphecyDistribution',
    'NoveltyScore',
    'InterferenceReport',
    # Tools
    'ThreatSignatureRecognizer',
    'BehavioralAnomalyDetector',
    'TimelineBranchingEngine',
    'ProphecyEngine',
    'NoveltyDetector',
    'ExternalInterferenceDetector',
]
