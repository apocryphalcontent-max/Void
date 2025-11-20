"""
Additional Phase 1 (MVP) Tool Implementations

This module implements the remaining Phase 1 tools:
- Pattern Prevalence Quantifier
- Local Entropy Microscope  
- Event Signature Classifier (stub)
"""

import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

from .base import AnalysisTool, MonitoringTool, Tool, ToolConfig
from .clock import Clock, get_clock
from .layered_tool import LayeredTool


class PatternPrevalenceQuantifier(LayeredTool, AnalysisTool):
    """
    Pattern Prevalence Quantifier - MVP Implementation

    Measures frequency and ubiquity of patterns across the system state corpus.
    Tracks pattern occurrences, contexts, and temporal stability.

    Phase: 1 (MVP)
    Layer: 2 (Analysis & Intelligence)
    Priority: P1 (High)
    Status: COMPLETE
    """

    _layer = 2
    _phase = 1

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        super().__init__(config)
        self._clock = clock or get_clock()
        self.pattern_counts: Counter = Counter()
        self.pattern_contexts: Dict[str, Set[str]] = defaultdict(set)
        self.pattern_first_seen: Dict[str, float] = {}
        self.pattern_last_seen: Dict[str, float] = {}
        self.total_observations = 0

    def initialize(self) -> bool:
        """
        Initialize the quantifier by resetting all pattern tracking state.

        This method clears all accumulated pattern statistics and prepares
        the quantifier for a fresh analysis session.

        Returns:
            bool: Always True, indicating successful initialization.
        """
        self.pattern_counts.clear()
        self.pattern_contexts.clear()
        self.pattern_first_seen.clear()
        self.pattern_last_seen.clear()
        self.total_observations = 0
        return True

    def shutdown(self) -> bool:
        """
        Clean up resources by clearing all pattern tracking data.

        This method releases memory used by pattern statistics. After shutdown,
        the quantifier can be reinitialized for reuse.

        Returns:
            bool: Always True, indicating successful shutdown.
        """
        self.pattern_counts.clear()
        self.pattern_contexts.clear()
        return True

    def suspend(self) -> bool:
        """
        Suspend tool operation while preserving state.

        The quantifier maintains all accumulated pattern data during suspension.

        Returns:
            bool: Always True, indicating successful suspension.
        """
        return True

    def resume(self) -> bool:
        """
        Resume tool operation from suspended state.

        The quantifier continues tracking patterns with all previous state intact.

        Returns:
            bool: Always True, indicating successful resumption.
        """
        return True

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze pattern prevalence in data
        
        Args:
            data: Data to analyze (dict with 'pattern' and 'context' keys)
        
        Returns:
            Dict with prevalence metrics
        """
        if not isinstance(data, dict):
            return {"error": "Data must be a dictionary"}

        pattern = data.get("pattern", "")
        context = data.get("context", "unknown")
        timestamp = data.get("timestamp", self._clock.now())

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
            "is_rare": frequency_ratio < 0.001,   # Less than 0.1% of observations
        }

    def _calculate_percentile(self, frequency: int) -> float:
        """
        Calculate the percentile rank of a given frequency.

        This method determines what percentage of patterns have frequencies
        less than or equal to the given frequency.

        Args:
            frequency: The frequency count to calculate percentile for.

        Returns:
            float: Percentile rank (0-100), where 100 means this is the
                  most frequent pattern.
        """
        if not self.pattern_counts:
            return 0.0

        frequencies = sorted(self.pattern_counts.values())
        position = sum(1 for f in frequencies if f <= frequency)
        return (position / len(frequencies)) * 100

    def get_top_patterns(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the N most prevalent patterns ranked by frequency.

        Args:
            n: Number of top patterns to return (default: 10).

        Returns:
            list: List of dictionaries containing pattern statistics:
                - pattern: The pattern identifier
                - frequency: Absolute occurrence count
                - frequency_ratio: Relative frequency (0-1)
                - context_diversity: Number of distinct contexts
        """
        results = []
        for pattern, count in self.pattern_counts.most_common(n):
            results.append({
                "pattern": pattern,
                "frequency": count,
                "frequency_ratio": count / self.total_observations,
                "context_diversity": len(self.pattern_contexts[pattern])
            })
        return results

    def get_rare_patterns(self, threshold: float = 0.001) -> List[Dict[str, Any]]:
        """
        Get patterns with frequency below a specified threshold.

        Identifies rare or anomalous patterns that occur infrequently.
        Useful for detecting outliers and novel events.

        Args:
            threshold: Maximum frequency ratio to consider rare (default: 0.001 = 0.1%).

        Returns:
            list: Sorted list (ascending frequency) of rare pattern statistics.
        """
        results = []
        for pattern, count in self.pattern_counts.items():
            ratio = count / self.total_observations if self.total_observations > 0 else 0
            if ratio < threshold:
                results.append({
                    "pattern": pattern,
                    "frequency": count,
                    "frequency_ratio": ratio
                })
        return sorted(results, key=lambda x: x["frequency"])

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Pattern Prevalence Quantifier",
            "category": "prevalence_novelty_quantifiers",
            "version": "1.0.0",
            "description": "Measures pattern frequency and ubiquity",
            "capabilities": {"pattern_tracking", "frequency_analysis", "context_diversity"},
            "dependencies": set(),
            "layer": 2,
            "phase": 1,
            "priority": "P1"
        }


class LocalEntropyMicroscope(LayeredTool, MonitoringTool):
    """
    Local Entropy Microscope - MVP Implementation

    Measures entropy at microscopic scales across system regions.
    Identifies entropy gradients, sources, and sinks.

    Phase: 1 (MVP)
    Layer: 2 (Analysis & Intelligence)
    Priority: P1 (High)
    Status: COMPLETE
    """

    _layer = 2
    _phase = 1

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        """
        Initialize the Local Entropy Microscope.

        Args:
            config: Tool configuration including resource quotas and parameters.
            clock: Optional clock instance for time tracking (defaults to system clock).
        """
        super().__init__(config)
        self._clock = clock or get_clock()
        self.region_states: Dict[str, List[Any]] = defaultdict(list)
        self.max_history = config.parameters.get('max_history', 1000)  # Keep last N states per region

    def initialize(self) -> bool:
        """
        Initialize the microscope by clearing all region state history.

        Returns:
            bool: Always True, indicating successful initialization.
        """
        self.region_states.clear()
        return True

    def shutdown(self) -> bool:
        """
        Clean up resources by clearing all region state data.

        Returns:
            bool: Always True, indicating successful shutdown.
        """
        self.region_states.clear()
        return True

    def suspend(self) -> bool:
        """
        Suspend tool operation while preserving all region state history.

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

    def on_event(self, event: Any):
        """
        Handle a monitoring event by observing the region state.

        This method is called by the monitoring framework when events occur.
        It processes events containing region and state information.

        Args:
            event: Event dictionary, expected to contain 'region' and 'state' keys.
        """
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
            self.region_states[region] = self.region_states[region][-self.max_history:]

        states = self.region_states[region]

        # Calculate local entropy
        entropy = self._calculate_shannon_entropy(states)

        # Calculate gradient (change in entropy)
        gradient = 0.0
        if len(states) >= 2:
            recent_entropy = self._calculate_shannon_entropy(states[-10:])
            older_entropy = self._calculate_shannon_entropy(states[-20:-10] if len(states) >= 20 else states[:-10])
            gradient = recent_entropy - older_entropy

        # Classify region
        is_source = gradient > 0.1  # Increasing entropy
        is_sink = gradient < -0.1   # Decreasing entropy
        is_stable = abs(gradient) < 0.05

        return {
            "region": region,
            "entropy": entropy,
            "gradient": gradient,
            "is_source": is_source,
            "is_sink": is_sink,
            "is_stable": is_stable,
            "observation_count": len(states),
            "unique_states": len(set(str(s) for s in states))
        }

    def _calculate_shannon_entropy(self, states: List[Any]) -> float:
        """
        Calculate Shannon entropy of state distribution.

        Uses the formula: H = -Σ p(x) * log2(p(x))
        where p(x) is the probability of each unique state.

        Args:
            states: List of state observations.

        Returns:
            float: Shannon entropy in bits. Higher values indicate more disorder/randomness.
                  0 means all states are identical (perfect order).
                  log2(n) means uniform distribution (maximum disorder).
        """
        if not states:
            return 0.0

        # Convert states to hashable representation
        hashable_states = []
        for s in states:
            try:
                # Try to use state as-is if it's already hashable
                hash(s)
                hashable_states.append(s)
            except TypeError:
                # For non-hashable states, use JSON serialization
                try:
                    hashable_states.append(json.dumps(s, sort_keys=True, default=str))
                except (TypeError, ValueError):
                    # Fallback to string representation for completely non-serializable objects
                    hashable_states.append(str(type(s).__name__) + str(id(s)))

        # Count occurrences
        counts = Counter(hashable_states)
        total = len(states)

        # Calculate entropy: H = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)

        return entropy

    def get_entropy_field(self) -> Dict[str, float]:
        """
        Get entropy values for all observed regions.

        Computes the current entropy for each region based on its state history.

        Returns:
            dict: Mapping from region name to entropy value (in bits).
        """
        field = {}
        for region, states in self.region_states.items():
            if states:
                field[region] = self._calculate_shannon_entropy(states)
        return field

    def identify_anomalies(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Identify regions with abnormal entropy levels.

        Detects regions where entropy deviates significantly from the mean.
        Useful for finding regions with unusual order (low entropy) or
        disorder (high entropy).

        Args:
            threshold: Minimum deviation (in bits) from mean to consider anomalous.

        Returns:
            list: Sorted list (by deviation, descending) of anomaly dictionaries:
                - region: Region identifier
                - entropy: Current entropy value
                - mean_entropy: Mean entropy across all regions
                - deviation: Absolute deviation from mean
                - type: "high" if above mean, "low" if below
        """
        field = self.get_entropy_field()
        if not field:
            return []

        mean_entropy = sum(field.values()) / len(field)
        anomalies = []

        for region, entropy in field.items():
            deviation = abs(entropy - mean_entropy)
            if deviation > threshold:
                anomalies.append({
                    "region": region,
                    "entropy": entropy,
                    "mean_entropy": mean_entropy,
                    "deviation": deviation,
                    "type": "high" if entropy > mean_entropy else "low"
                })

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
            "priority": "P1"
        }


class EventSignatureClassifier(LayeredTool, Tool):
    """
    Event Signature Classifier - MVP Implementation with Naive Bayes

    Classifies events into taxonomic classes using a Naive Bayes classifier.
    Learns from labeled training data and provides probabilistic classifications.

    Phase: 1 (MVP)
    Layer: 1 (Sensing & Instrumentation)
    Priority: P1 (High)
    Status: COMPLETE

    The classifier uses a simple Naive Bayes model that:
    - Extracts features from event dictionaries
    - Learns class probabilities from training examples
    - Provides confidence scores based on posterior probabilities
    """

    _layer = 1
    _phase = 1

    def __init__(self, config: ToolConfig, clock: Optional[Clock] = None):
        """
        Initialize the Event Signature Classifier.

        Args:
            config: Tool configuration including parameters:
                - smoothing_alpha: Laplace smoothing parameter (default: 1.0)
            clock: Optional clock instance for time tracking (defaults to system clock).
        """
        super().__init__(config)
        self._clock = clock or get_clock()

        # Known event classes
        self.event_classes: Set[str] = {
            "memory_allocation",
            "memory_deallocation",
            "execution_start",
            "execution_end",
            "io_read",
            "io_write",
            "exception",
            "gc_cycle",
            "network_send",
            "network_receive",
            "unknown"
        }

        # Naive Bayes model parameters
        self.class_priors: Dict[str, float] = {}  # P(class)
        self.feature_likelihoods: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # P(feature|class)
        self.feature_vocabulary: Set[str] = set()
        self.training_count: int = 0
        self.class_counts: Counter = Counter()

        # Smoothing parameter (Laplace smoothing)
        self.smoothing_alpha = config.parameters.get('smoothing_alpha', 1.0)

        # Feature extraction thresholds
        self.size_small_threshold = config.parameters.get('size_small_threshold', 100)
        self.size_medium_threshold = config.parameters.get('size_medium_threshold', 10000)

        # Classification history
        self.classification_history: List[Dict[str, Any]] = []
        self.max_history = config.parameters.get('max_history', 10000)

        # Is model trained?
        self._is_trained = False

    def initialize(self) -> bool:
        """
        Initialize the classifier with default training data.

        Sets up the classifier with some basic training examples to bootstrap
        the model. Additional training can be done via train().

        Returns:
            bool: Always True, indicating successful initialization.
        """
        self.classification_history.clear()

        # Bootstrap with some basic training examples
        training_data = [
            ({"type": "alloc", "size": 1024}, "memory_allocation"),
            ({"type": "dealloc", "address": "0x1234"}, "memory_deallocation"),
            ({"type": "start", "function": "main"}, "execution_start"),
            ({"type": "end", "function": "main"}, "execution_end"),
            ({"type": "read", "fd": 0}, "io_read"),
            ({"type": "write", "fd": 1}, "io_write"),
            ({"type": "error", "exception": "ValueError"}, "exception"),
            ({"type": "gc", "collected": 100}, "gc_cycle"),
            ({"type": "send", "bytes": 512}, "network_send"),
            ({"type": "recv", "bytes": 256}, "network_receive"),
        ]

        self.train(training_data)
        return True

    def shutdown(self) -> bool:
        """
        Clean up resources by clearing classification history.

        The trained model is preserved for potential reuse.

        Returns:
            bool: Always True, indicating successful shutdown.
        """
        self.classification_history.clear()
        return True

    def suspend(self) -> bool:
        """
        Suspend tool operation while preserving trained model.

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

    def train(self, training_data: List[tuple]) -> None:
        """
        Train the Naive Bayes classifier on labeled examples.

        Args:
            training_data: List of (event, label) tuples where:
                - event is a dictionary containing event data
                - label is a string class name
        """
        # Reset model
        self.class_counts.clear()
        self.feature_vocabulary.clear()
        self.feature_likelihoods.clear()

        # Count class occurrences and extract features
        feature_counts = defaultdict(lambda: defaultdict(Counter))  # feature_counts[class][feature] = count

        for event, label in training_data:
            self.class_counts[label] += 1
            features = self._extract_features(event)

            for feature in features:
                self.feature_vocabulary.add(feature)
                feature_counts[label][feature][True] += 1

            self.training_count += 1

        # Calculate class priors: P(class)
        for cls, count in self.class_counts.items():
            self.class_priors[cls] = count / self.training_count

        # Calculate feature likelihoods: P(feature|class) with Laplace smoothing
        for cls in self.class_counts.keys():
            class_feature_count = sum(feature_counts[cls][f][True] for f in self.feature_vocabulary if f in feature_counts[cls])

            for feature in self.feature_vocabulary:
                feature_count = feature_counts[cls][feature][True] if feature in feature_counts[cls] else 0

                # Laplace smoothing
                likelihood = (feature_count + self.smoothing_alpha) / (
                    class_feature_count + self.smoothing_alpha * len(self.feature_vocabulary)
                )

                self.feature_likelihoods[cls][feature] = likelihood

        self._is_trained = True

    def classify_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an event using the trained Naive Bayes model.

        Uses Bayes' theorem to compute posterior probabilities:
        P(class|features) ∝ P(class) * ∏ P(feature|class)

        Args:
            event: Event dictionary containing event data.

        Returns:
            dict: Classification result containing:
                - event_id: Event identifier (if provided)
                - classification: Predicted class label
                - confidence: Probability of predicted class (0-1)
                - probabilities: Dict of all class probabilities
                - features: Extracted features
                - timestamp: Classification timestamp
        """
        if not self._is_trained:
            # Fallback to rule-based for untrained model
            return self._rule_based_classify(event)

        features = self._extract_features(event)

        # Calculate posterior probabilities for each class
        posteriors = {}
        for cls in self.class_priors.keys():
            # Start with prior: P(class)
            log_prob = math.log(self.class_priors[cls])

            # Multiply by likelihoods: P(feature|class)
            for feature in features:
                if feature in self.feature_likelihoods[cls]:
                    log_prob += math.log(self.feature_likelihoods[cls][feature])
                else:
                    # Unseen feature - use smoothing
                    log_prob += math.log(self.smoothing_alpha / (
                        sum(self.class_counts.values()) + self.smoothing_alpha * len(self.feature_vocabulary)
                    ))

            posteriors[cls] = log_prob

        # Normalize to get probabilities
        max_log_prob = max(posteriors.values())
        exp_probs = {cls: math.exp(log_prob - max_log_prob) for cls, log_prob in posteriors.items()}
        total = sum(exp_probs.values())
        probabilities = {cls: prob / total for cls, prob in exp_probs.items()}

        # Get most likely class
        classification = max(probabilities.items(), key=lambda x: x[1])

        result = {
            "event_id": event.get("id", ""),
            "classification": classification[0],
            "confidence": classification[1],
            "probabilities": probabilities,
            "features": features,
            "timestamp": self._clock.now()
        }

        # Store in history
        self.classification_history.append(result)
        if len(self.classification_history) > self.max_history:
            self.classification_history = self.classification_history[-self.max_history:]

        return result

    def _rule_based_classify(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback rule-based classification for untrained model.

        Args:
            event: Event dictionary.

        Returns:
            dict: Classification result.
        """
        event_type = event.get("type", "unknown")

        # Simple rule-based mapping
        type_mapping = {
            "alloc": "memory_allocation",
            "dealloc": "memory_deallocation",
            "start": "execution_start",
            "end": "execution_end",
            "read": "io_read",
            "write": "io_write",
            "error": "exception",
            "exception": "exception",
            "gc": "gc_cycle",
            "send": "network_send",
            "recv": "network_receive",
        }

        classification = type_mapping.get(event_type, "unknown")
        confidence = 1.0 if event_type in type_mapping else 0.0

        result = {
            "event_id": event.get("id", ""),
            "classification": classification,
            "confidence": confidence,
            "probabilities": {classification: confidence},
            "features": self._extract_features(event),
            "timestamp": self._clock.now()
        }

        self.classification_history.append(result)
        return result

    def _extract_features(self, event: Dict[str, Any]) -> List[str]:
        """
        Extract features from an event for classification.

        Features include:
        - Event type
        - Presence of specific keys (size, error, address, etc.)
        - Value patterns for numeric fields

        Args:
            event: Event dictionary.

        Returns:
            list: List of feature strings.
        """
        features = []

        # Type-based features
        if "type" in event:
            features.append(f"type:{event['type']}")

        # Key presence features
        for key in ["size", "error", "exception", "address", "fd", "function", "bytes", "collected"]:
            if key in event:
                features.append(f"has_{key}")

        # Value-based features
        if "size" in event and isinstance(event["size"], (int, float)):
            if event["size"] < self.size_small_threshold:
                features.append("size_small")
            elif event["size"] < self.size_medium_threshold:
                features.append("size_medium")
            else:
                features.append("size_large")

        if "fd" in event:
            fd = event["fd"]
            if fd == 0:
                features.append("fd_stdin")
            elif fd == 1:
                features.append("fd_stdout")
            elif fd == 2:
                features.append("fd_stderr")
            else:
                features.append("fd_other")

        return features

    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get statistics on classification history.

        Returns:
            dict: Statistics including:
                - total: Total number of classifications
                - by_class: Count of classifications per class
                - average_confidence: Mean confidence score
                - model_trained: Whether model has been trained
                - training_examples: Number of training examples
        """
        if not self.classification_history:
            return {
                "total": 0,
                "by_class": {},
                "average_confidence": 0.0,
                "model_trained": self._is_trained,
                "training_examples": self.training_count
            }

        class_counts = Counter(c["classification"] for c in self.classification_history)

        return {
            "total": len(self.classification_history),
            "by_class": dict(class_counts),
            "average_confidence": sum(c["confidence"] for c in self.classification_history) / len(self.classification_history),
            "model_trained": self._is_trained,
            "training_examples": self.training_count,
            "feature_vocabulary_size": len(self.feature_vocabulary)
        }

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get classifier metadata.

        Returns:
            dict: Metadata including name, version, capabilities, etc.
        """
        return {
            "name": "Event Signature Classifier",
            "category": "anomaly_event_classifiers",
            "version": "1.0.0",
            "description": "Naive Bayes classifier for event categorization",
            "capabilities": {"event_classification", "feature_extraction", "probabilistic_inference"},
            "dependencies": set(),
            "layer": 1,
            "phase": 1,
            "priority": "P1",
            "status": "complete",
            "algorithm": "naive_bayes"
        }


# Export new tools
__all__ = [
    "PatternPrevalenceQuantifier",
    "LocalEntropyMicroscope",
    "EventSignatureClassifier"
]
