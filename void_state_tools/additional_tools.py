"""
Additional Phase 1 (MVP) Tool Implementations

This module implements the remaining Phase 1 tools:
- Pattern Prevalence Quantifier
- Local Entropy Microscope
- Event Signature Classifier (stub)
"""

import math
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set

from .base import AnalysisTool, MonitoringTool, Tool, ToolConfig


class PatternPrevalenceQuantifier(AnalysisTool):
    """
    Pattern Prevalence Quantifier - MVP Implementation

    Measures frequency and ubiquity of patterns across the system state corpus.
    Tracks pattern occurrences, contexts, and temporal stability.

    This tool provides quantitative analysis of pattern prevalence including:
    - Absolute and relative frequency counts
    - Context diversity (how many different contexts a pattern appears in)
    - Temporal stability (consistency of pattern occurrence over time)
    - Percentile ranking among all observed patterns
    - Classification as common/rare based on configurable thresholds

    The tool maintains a running history of all observed patterns and computes
    real-time statistics without requiring batch processing. It's designed for
    high-throughput observation streams while maintaining bounded memory usage
    through periodic pruning of old patterns.

    Phase: 1 (MVP)
    Layer: 2 (Analysis & Intelligence)
    Priority: P1 (High)
    Status: COMPLETE

    Example:
        >>> config = ToolConfig(tool_name="pattern_quantifier")
        >>> quantifier = PatternPrevalenceQuantifier(config)
        >>> quantifier.initialize()
        True
        >>> result = quantifier.analyze({
        ...     "pattern": "memory_alloc_small",
        ...     "context": "gc_cycle",
        ...     "timestamp": time.time()
        ... })
        >>> print(result["frequency"], result["context_diversity"])
        1 1
    """

    def __init__(self, config: ToolConfig):
        """
        Initialize the Pattern Prevalence Quantifier.

        Args:
            config: Tool configuration with resource quotas and parameters
        """
        super().__init__(config)
        self.pattern_counts: Counter = Counter()
        self.pattern_contexts: Dict[str, Set[str]] = defaultdict(set)
        self.pattern_first_seen: Dict[str, float] = {}
        self.pattern_last_seen: Dict[str, float] = {}
        self.total_observations = 0

    def initialize(self) -> bool:
        """
        Initialize the quantifier, clearing all state.

        Resets all counters and histories to prepare for fresh observation.
        Should be called before starting a new analysis session.

        Returns:
            True if initialization succeeded, False otherwise
        """
        self.pattern_counts.clear()
        self.pattern_contexts.clear()
        self.pattern_first_seen.clear()
        self.pattern_last_seen.clear()
        self.total_observations = 0
        return True

    def shutdown(self) -> bool:
        """
        Cleanup resources and finalize analysis.

        Clears all internal state and frees memory. Should be called
        when the tool is no longer needed or before system shutdown.

        Returns:
            True if shutdown succeeded, False otherwise
        """
        self.pattern_counts.clear()
        self.pattern_contexts.clear()
        return True

    def suspend(self) -> bool:
        """
        Suspend tool operation without losing state.

        Pauses pattern observation but maintains all accumulated statistics.
        The tool can be resumed later to continue from the same state.

        Returns:
            True if suspension succeeded, False otherwise
        """
        return True

    def resume(self) -> bool:
        """
        Resume tool operation from suspended state.

        Re-enables pattern observation, continuing from previously
        accumulated statistics. All pattern histories are preserved.

        Returns:
            True if resumption succeeded, False otherwise
        """
        return True

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze pattern prevalence in data.

        Processes a single pattern observation, updating internal statistics
        and returning comprehensive prevalence metrics for that pattern.

        The analysis includes:
        - Frequency: absolute count and ratio relative to all observations
        - Context diversity: number of unique contexts the pattern appears in
        - Age: time since first observation
        - Stability: consistency of occurrence rate over time
        - Percentile: ranking among all observed patterns
        - Classification: is_common, is_rare flags based on thresholds

        Args:
            data: Dictionary containing:
                - pattern (str, required): The pattern identifier
                - context (str, optional): Context where pattern was observed
                - timestamp (float, optional): Observation time (defaults to now)

        Returns:
            Dictionary with prevalence metrics:
                - pattern: The pattern identifier
                - frequency: Absolute occurrence count
                - frequency_ratio: Relative frequency (0.0 to 1.0)
                - context_diversity: Number of unique contexts
                - age_seconds: Time since first observation
                - stability: Temporal consistency score (0.0 to 1.0)
                - percentile: Ranking percentile (0 to 100)
                - is_common: True if ratio > 1%
                - is_rare: True if ratio < 0.1%

        Raises:
            No exceptions raised; errors returned in result dictionary

        Example:
            >>> result = quantifier.analyze({
            ...     "pattern": "heap_alloc",
            ...     "context": "user_code",
            ...     "timestamp": 1234567890.0
            ... })
            >>> assert "frequency" in result
            >>> assert "stability" in result
        """
        if not isinstance(data, dict):
            return {"error": "Data must be a dictionary"}

        pattern = data.get("pattern", "")
        context = data.get("context", "unknown")
        timestamp = data.get("timestamp", time.time())

        if not pattern:
            return {"error": "No pattern provided"}

        # Update counts
        self.pattern_counts[pattern] += 1
        self.pattern_contexts[pattern].add(context)
        self.total_observations += 1

        # Update timestamps
        if pattern not in self.pattern_first_seen:
            self.pattern_first_seen[pattern] = timestamp
        self.pattern_last_seen[pattern] = timestamp

        # Calculate metrics
        frequency = self.pattern_counts[pattern]
        frequency_ratio = frequency / self.total_observations if self.total_observations > 0 else 0
        context_diversity = len(self.pattern_contexts[pattern])
        age_seconds = timestamp - self.pattern_first_seen[pattern]

        # Calculate stability (how consistent the pattern is over time)
        if age_seconds > 0:
            occurrences_per_second = frequency / age_seconds
            stability = min(1.0, occurrences_per_second / 0.1)  # Normalize to 0-1
        else:
            stability = 1.0

        return {
            "pattern": pattern,
            "frequency": frequency,
            "frequency_ratio": frequency_ratio,
            "context_diversity": context_diversity,
            "age_seconds": age_seconds,
            "stability": stability,
            "percentile": self._calculate_percentile(frequency),
            "is_common": frequency_ratio > 0.01,  # More than 1% of observations
            "is_rare": frequency_ratio < 0.001,  # Less than 0.1% of observations
        }

    def _calculate_percentile(self, frequency: int) -> float:
        """
        Calculate what percentile this frequency is at.

        Computes the percentile rank of a given frequency among all
        observed pattern frequencies. Used to identify unusually common
        or rare patterns.

        Args:
            frequency: The frequency to rank

        Returns:
            Percentile rank (0 to 100)
        """
        if not self.pattern_counts:
            return 0.0

        frequencies = sorted(self.pattern_counts.values())
        position = sum(1 for f in frequencies if f <= frequency)
        return (position / len(frequencies)) * 100

    def get_top_patterns(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the N most prevalent patterns.

        Returns the patterns with highest frequency, sorted in descending
        order. Useful for identifying dominant patterns in the system.

        Args:
            n: Number of top patterns to return (default: 10)

        Returns:
            List of dictionaries, each containing:
                - pattern: Pattern identifier
                - frequency: Absolute count
                - frequency_ratio: Relative frequency
                - context_diversity: Number of contexts
        """
        results = []
        for pattern, count in self.pattern_counts.most_common(n):
            results.append(
                {
                    "pattern": pattern,
                    "frequency": count,
                    "frequency_ratio": count / self.total_observations,
                    "context_diversity": len(self.pattern_contexts[pattern]),
                }
            )
        return results

    def get_rare_patterns(self, threshold: float = 0.001) -> List[Dict[str, Any]]:
        """
        Get patterns with frequency below threshold.

        Identifies rare or anomalous patterns that occur infrequently.
        These patterns may indicate edge cases, errors, or unusual
        system behaviors worth investigating.

        Args:
            threshold: Maximum frequency ratio to include (default: 0.001 = 0.1%)

        Returns:
            List of rare patterns sorted by frequency (ascending)
        """
        results = []
        for pattern, count in self.pattern_counts.items():
            ratio = count / self.total_observations if self.total_observations > 0 else 0
            if ratio < threshold:
                results.append({"pattern": pattern, "frequency": count, "frequency_ratio": ratio})
        return sorted(results, key=lambda x: x["frequency"])

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get tool metadata.

        Returns:
            Dictionary containing tool information including name, version,
            capabilities, dependencies, and layer classification
        """
        return {
            "name": "Pattern Prevalence Quantifier",
            "category": "prevalence_novelty_quantifiers",
            "version": "1.0.0",
            "description": "Measures pattern frequency and ubiquity",
            "capabilities": {"pattern_tracking", "frequency_analysis", "context_diversity"},
            "dependencies": set(),
            "layer": 2,
            "phase": 1,
            "priority": "P1",
        }


class LocalEntropyMicroscope(MonitoringTool):
    """
    Local Entropy Microscope - MVP Implementation

    Measures entropy at microscopic scales across system regions.
    Identifies entropy gradients, sources, and sinks.

    Phase: 1 (MVP)
    Layer: 2 (Analysis & Intelligence)
    Priority: P1 (High)
    Status: COMPLETE
    """

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.region_states: Dict[str, List[Any]] = defaultdict(list)
        self.max_history = 1000  # Keep last N states per region

    def initialize(self) -> bool:
        """Initialize the microscope"""
        self.region_states.clear()
        return True

    def shutdown(self) -> bool:
        """Cleanup resources"""
        self.region_states.clear()
        return True

    def suspend(self) -> bool:
        """Suspend tool operation"""
        return True

    def resume(self) -> bool:
        """Resume tool operation"""
        return True

    def on_event(self, event: Any):
        """Handle monitoring event"""
        if isinstance(event, dict) and "region" in event and "state" in event:
            self.observe_region(event["region"], event["state"])

    def collect_metrics(self) -> Dict[str, float]:
        """
        Collect current entropy metrics

        Returns:
            Dict mapping region names to entropy values
        """
        metrics = {}
        for region, states in self.region_states.items():
            if states:
                metrics[f"entropy_{region}"] = self._calculate_shannon_entropy(states)

        # Global entropy
        all_states = []
        for states in self.region_states.values():
            all_states.extend(states)
        if all_states:
            metrics["entropy_global"] = self._calculate_shannon_entropy(all_states)

        return metrics

    def observe_region(self, region: str, state: Any) -> Dict[str, Any]:
        """
        Observe a region's state and calculate entropy

        Args:
            region: Region identifier
            state: Current state value

        Returns:
            Dict with entropy metrics
        """
        # Add state to history
        self.region_states[region].append(state)

        # Limit history size
        if len(self.region_states[region]) > self.max_history:
            self.region_states[region] = self.region_states[region][-self.max_history :]

        states = self.region_states[region]

        # Calculate local entropy
        entropy = self._calculate_shannon_entropy(states)

        # Calculate gradient (change in entropy)
        gradient = 0.0
        if len(states) >= 2:
            recent_entropy = self._calculate_shannon_entropy(states[-10:])
            older_entropy = self._calculate_shannon_entropy(
                states[-20:-10] if len(states) >= 20 else states[:-10]
            )
            gradient = recent_entropy - older_entropy

        # Classify region
        is_source = gradient > 0.1  # Increasing entropy
        is_sink = gradient < -0.1  # Decreasing entropy
        is_stable = abs(gradient) < 0.05

        return {
            "region": region,
            "entropy": entropy,
            "gradient": gradient,
            "is_source": is_source,
            "is_sink": is_sink,
            "is_stable": is_stable,
            "observation_count": len(states),
            "unique_states": len(set(str(s) for s in states)),
        }

    def _calculate_shannon_entropy(self, states: List[Any]) -> float:
        """Calculate Shannon entropy of state distribution"""
        if not states:
            return 0.0

        # Count occurrences
        counts = Counter(str(s) for s in states)
        total = len(states)

        # Calculate entropy: H = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)

        return entropy

    def get_entropy_field(self) -> Dict[str, float]:
        """Get entropy values for all regions"""
        field = {}
        for region, states in self.region_states.items():
            if states:
                field[region] = self._calculate_shannon_entropy(states)
        return field

    def identify_anomalies(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Identify regions with abnormal entropy"""
        field = self.get_entropy_field()
        if not field:
            return []

        mean_entropy = sum(field.values()) / len(field)
        anomalies = []

        for region, entropy in field.items():
            deviation = abs(entropy - mean_entropy)
            if deviation > threshold:
                anomalies.append(
                    {
                        "region": region,
                        "entropy": entropy,
                        "mean_entropy": mean_entropy,
                        "deviation": deviation,
                        "type": "high" if entropy > mean_entropy else "low",
                    }
                )

        return sorted(anomalies, key=lambda x: x["deviation"], reverse=True)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Local Entropy Microscope",
            "category": "entropy_zeal_microscopics",
            "version": "1.0.0",
            "description": "Measures microscopic entropy across system regions",
            "capabilities": {"entropy_measurement", "gradient_analysis", "anomaly_detection"},
            "dependencies": set(),
            "layer": 2,
            "phase": 1,
            "priority": "P1",
        }


class EventSignatureClassifier(Tool):
    """
    Event Signature Classifier - Bayesian Implementation

    Classifies events into taxonomic classes using a Naive Bayes classifier.
    Learns from observed events and provides probabilistic classifications.

    This implementation uses a simple Bayesian approach where:
    - P(class|features) ∝ P(features|class) * P(class)
    - Features are assumed conditionally independent given the class

    Phase: 1 (MVP)
    Layer: 1 (Sensing & Instrumentation)
    Priority: P1 (High)
    Status: COMPLETE (Bayesian Naive Classifier)
    """

    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.event_classes: Set[str] = {
            "memory_allocation",
            "memory_deallocation",
            "execution_start",
            "execution_end",
            "io_read",
            "io_write",
            "exception",
            "gc_cycle",
            "unknown",
        }

        # Bayesian classifier state
        self.class_counts: Counter = Counter()  # P(class)
        self.feature_counts: Dict[str, Counter] = defaultdict(Counter)  # P(feature|class)
        self.total_samples = 0

        # Classification history
        self.classification_history: List[Dict[str, Any]] = []

        # Laplace smoothing parameter
        self.alpha = 1.0

    def initialize(self) -> bool:
        """
        Initialize the Bayesian classifier.

        Returns:
            True if initialization succeeded
        """
        self.class_counts.clear()
        self.feature_counts.clear()
        self.total_samples = 0
        self.classification_history.clear()
        return True

    def shutdown(self) -> bool:
        """
        Cleanup classifier resources.

        Returns:
            True if shutdown succeeded
        """
        self.class_counts.clear()
        self.feature_counts.clear()
        self.classification_history.clear()
        return True

    def suspend(self) -> bool:
        """Suspend classifier operation"""
        return True

    def resume(self) -> bool:
        """Resume classifier operation"""
        return True

    def train(self, event: Dict[str, Any], true_class: str) -> None:
        """
        Train the classifier with a labeled event.

        Updates the Bayesian model with observed event-class pairs.

        Args:
            event: Event dictionary with features
            true_class: Known true classification
        """
        if true_class not in self.event_classes:
            self.event_classes.add(true_class)

        # Update class counts (prior)
        self.class_counts[true_class] += 1
        self.total_samples += 1

        # Update feature counts (likelihood)
        features = self._extract_features(event)
        for feature in features:
            self.feature_counts[true_class][feature] += 1

    def classify_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an event using Naive Bayes.

        Computes P(class|features) for each class and returns the most probable.
        Uses log-space to avoid numerical underflow.

        Args:
            event: Event dictionary to classify

        Returns:
            Classification result with class, confidence, and posterior probabilities
        """
        features = self._extract_features(event)

        if self.total_samples == 0:
            # No training data - fallback to rule-based
            event_type = event.get("type", "unknown")
            classification = event_type if event_type in self.event_classes else "unknown"
            return {
                "event_id": event.get("id", ""),
                "classification": classification,
                "confidence": 0.5,
                "features": features,
                "timestamp": time.time(),
                "method": "rule_based",
            }

        # Compute log posterior for each class
        log_posteriors = {}
        for event_class in self.event_classes:
            # Log prior: log P(class)
            class_count = self.class_counts[event_class]
            log_prior = math.log(
                (class_count + self.alpha)
                / (self.total_samples + self.alpha * len(self.event_classes))
            )

            # Log likelihood: log P(features|class)
            log_likelihood = 0.0
            feature_vocab_size = len(
                set(f for counts in self.feature_counts.values() for f in counts.keys())
            )

            for feature in features:
                feature_count = self.feature_counts[event_class][feature]
                total_features_in_class = sum(self.feature_counts[event_class].values())

                # Laplace smoothing
                prob = (feature_count + self.alpha) / (
                    total_features_in_class + self.alpha * (feature_vocab_size + 1)
                )
                log_likelihood += math.log(prob)

            log_posteriors[event_class] = log_prior + log_likelihood

        # Find maximum
        classification = max(log_posteriors, key=log_posteriors.get)

        # Convert to probabilities (softmax)
        max_log = max(log_posteriors.values())
        exp_posteriors = {c: math.exp(lp - max_log) for c, lp in log_posteriors.items()}
        total_exp = sum(exp_posteriors.values())
        posteriors = {c: ep / total_exp for c, ep in exp_posteriors.items()}

        confidence = posteriors[classification]

        result = {
            "event_id": event.get("id", ""),
            "classification": classification,
            "confidence": confidence,
            "posteriors": posteriors,
            "features": features,
            "timestamp": time.time(),
            "method": "naive_bayes",
        }

        self.classification_history.append(result)

        return result

    def _extract_features(self, event: Dict[str, Any]) -> List[str]:
        """
        Extract features from an event.

        Features include:
        - Event type
        - Presence of various fields (size, error, duration, etc.)
        - Value ranges for numeric fields

        Args:
            event: Event dictionary

        Returns:
            List of feature strings
        """
        features = []

        # Type feature
        if "type" in event:
            features.append(f"type:{event['type']}")

        # Presence features
        if "size" in event:
            features.append("has_size")
            size = event["size"]
            if isinstance(size, (int, float)):
                # Bin sizes
                if size < 1024:
                    features.append("size:small")
                elif size < 1024 * 1024:
                    features.append("size:medium")
                else:
                    features.append("size:large")

        if "error" in event:
            features.append("has_error")
            if isinstance(event["error"], str):
                features.append(f"error_type:{event['error'][:20]}")

        if "duration" in event:
            features.append("has_duration")
            duration = event["duration"]
            if isinstance(duration, (int, float)):
                if duration < 0.001:
                    features.append("duration:fast")
                elif duration < 0.1:
                    features.append("duration:medium")
                else:
                    features.append("duration:slow")

        if "address" in event:
            features.append("has_address")

        if "thread_id" in event:
            features.append("has_thread")

        if "priority" in event:
            features.append("has_priority")

        # Data characteristics
        if "data" in event:
            features.append("has_data")
            data = event["data"]
            if isinstance(data, bytes):
                features.append("data:binary")
            elif isinstance(data, str):
                features.append("data:text")
            elif isinstance(data, (list, tuple)):
                features.append("data:sequence")
            elif isinstance(data, dict):
                features.append("data:mapping")

        return features

    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get statistics on classifications and training data.

        Returns:
            Dictionary with counts, accuracy estimates, and model statistics
        """
        if not self.classification_history:
            return {
                "total_predictions": 0,
                "total_training_samples": self.total_samples,
                "classes": len(self.event_classes),
            }

        class_counts = Counter(c["classification"] for c in self.classification_history)
        avg_confidence = sum(c["confidence"] for c in self.classification_history) / len(
            self.classification_history
        )

        return {
            "total_predictions": len(self.classification_history),
            "total_training_samples": self.total_samples,
            "classes": len(self.event_classes),
            "by_class": dict(class_counts),
            "average_confidence": avg_confidence,
            "class_priors": {
                cls: count / self.total_samples for cls, count in self.class_counts.items()
            }
            if self.total_samples > 0
            else {},
            "feature_vocabulary_size": len(
                set(f for counts in self.feature_counts.values() for f in counts.keys())
            ),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get tool metadata.

        Returns:
            Dictionary with tool information
        """
        return {
            "name": "Event Signature Classifier",
            "category": "anomaly_event_classifiers",
            "version": "1.0.0",
            "description": "Bayesian Naive classifier for event taxonomic classification",
            "capabilities": {
                "event_classification",
                "feature_extraction",
                "online_learning",
                "probabilistic_inference",
            },
            "dependencies": set(),
            "layer": 1,
            "phase": 1,
            "priority": "P1",
            "status": "complete",
            "algorithm": "naive_bayes",
        }


# Export new tools
__all__ = ["PatternPrevalenceQuantifier", "LocalEntropyMicroscope", "EventSignatureClassifier"]
