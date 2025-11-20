"""
Additional Phase 1 (MVP) Tool Implementations

This module implements the remaining Phase 1 tools:
- Pattern Prevalence Quantifier
- Local Entropy Microscope  
- Event Signature Classifier (stub)
"""

from typing import Dict, Any, Set, List, Optional
from collections import defaultdict, Counter
import math
import time

from void_state_tools.base import (
    Tool, ToolConfig, AnalysisTool, MonitoringTool
)


class PatternPrevalenceQuantifier(AnalysisTool):
    """
    Pattern Prevalence Quantifier - MVP Implementation
    
    Measures frequency and ubiquity of patterns across the system state corpus.
    Tracks pattern occurrences, contexts, and temporal stability.
    
    Phase: 1 (MVP)
    Layer: 2 (Analysis & Intelligence)
    Priority: P1 (High)
    Status: COMPLETE
    """
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.pattern_counts: Counter = Counter()
        self.pattern_contexts: Dict[str, Set[str]] = defaultdict(set)
        self.pattern_first_seen: Dict[str, float] = {}
        self.pattern_last_seen: Dict[str, float] = {}
        self.total_observations = 0
    
    def initialize(self) -> bool:
        """Initialize the quantifier"""
        self.pattern_counts.clear()
        self.pattern_contexts.clear()
        self.pattern_first_seen.clear()
        self.pattern_last_seen.clear()
        self.total_observations = 0
        return True
    
    def shutdown(self) -> bool:
        """Cleanup resources"""
        self.pattern_counts.clear()
        self.pattern_contexts.clear()
        return True
    
    def suspend(self) -> bool:
        """Suspend tool operation"""
        return True
    
    def resume(self) -> bool:
        """Resume tool operation"""
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
            "is_rare": frequency_ratio < 0.001,   # Less than 0.1% of observations
        }
    
    def _calculate_percentile(self, frequency: int) -> float:
        """Calculate what percentile this frequency is at"""
        if not self.pattern_counts:
            return 0.0
        
        frequencies = sorted(self.pattern_counts.values())
        position = sum(1 for f in frequencies if f <= frequency)
        return (position / len(frequencies)) * 100
    
    def get_top_patterns(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the N most prevalent patterns"""
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
        """Get patterns with frequency below threshold"""
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


class EventSignatureClassifier(Tool):
    """
    Event Signature Classifier - MVP Stub
    
    Classifies events into taxonomic classes based on features.
    
    Phase: 1 (MVP)
    Layer: 1 (Sensing & Instrumentation)
    Priority: P1 (High)
    Status: PLANNED (Basic implementation)
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
            "unknown"
        }
        self.classification_history: List[Dict[str, Any]] = []
    
    def initialize(self) -> bool:
        """Initialize classifier"""
        self.classification_history.clear()
        return True
    
    def shutdown(self) -> bool:
        """Cleanup resources"""
        self.classification_history.clear()
        return True
    
    def suspend(self) -> bool:
        """Suspend tool operation"""
        return True
    
    def resume(self) -> bool:
        """Resume tool operation"""
        return True
    
    def classify_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an event
        
        Args:
            event: Event dictionary with 'type', 'data', etc.
        
        Returns:
            Classification result
        """
        event_type = event.get("type", "unknown")
        
        # Simple rule-based classification (placeholder for ML)
        if event_type in self.event_classes:
            classification = event_type
            confidence = 1.0
        else:
            classification = "unknown"
            confidence = 0.0
        
        result = {
            "event_id": event.get("id", ""),
            "classification": classification,
            "confidence": confidence,
            "features": self._extract_features(event),
            "timestamp": time.time()
        }
        
        self.classification_history.append(result)
        
        return result
    
    def _extract_features(self, event: Dict[str, Any]) -> List[str]:
        """Extract features from event (placeholder)"""
        features = []
        if "type" in event:
            features.append(f"type:{event['type']}")
        if "size" in event:
            features.append(f"has_size")
        if "error" in event:
            features.append(f"has_error")
        return features
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics on classifications"""
        if not self.classification_history:
            return {"total": 0}
        
        class_counts = Counter(c["classification"] for c in self.classification_history)
        
        return {
            "total": len(self.classification_history),
            "by_class": dict(class_counts),
            "average_confidence": sum(c["confidence"] for c in self.classification_history) / len(self.classification_history)
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Event Signature Classifier",
            "category": "anomaly_event_classifiers",
            "version": "1.0.0-stub",
            "description": "Classifies events into taxonomic classes",
            "capabilities": {"event_classification", "feature_extraction"},
            "dependencies": set(),
            "layer": 1,
            "phase": 1,
            "priority": "P1",
            "status": "basic_implementation"
        }


# Export new tools
__all__ = [
    "PatternPrevalenceQuantifier",
    "LocalEntropyMicroscope",
    "EventSignatureClassifier"
]
