"""
Emergent Behavior Detection for Void-State

Detects complex emergent patterns, phase transitions, criticality,
and self-organization in AI agent systems.

Based on complexity science, synergetics, and non-linear dynamics.

Author: Void-State Research Team
Version: 3.1.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import math


@dataclass
class EmergentPattern:
    """Detected emergent behavior pattern."""
    pattern_id: str
    pattern_type: str  # 'synchronization', 'criticality', 'self-organization', etc.
    strength: float    # [0, 1] confidence
    components: List[str]  # Participating components
    emergence_time: int
    description: str
    

class SynchronizationDetector:
    """
    Detect synchronization and coordination across system components.
    
    Based on Kuramoto model and phase coupling.
    """
    
    def __init__(self, coupling_threshold: float = 0.7):
        self.coupling_threshold = coupling_threshold
        self.phase_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def add_oscillator_state(self, component_id: str, phase: float):
        """
        Add phase information for an oscillator.
        
        Args:
            component_id: Component identifier
            phase: Phase angle [0, 2π]
        """
        self.phase_history[component_id].append(phase % (2 * math.pi))
    
    def compute_order_parameter(self) -> float:
        """
        Compute Kuramoto order parameter r.
        
        r = |⟨e^(iθ)⟩| measures synchronization level
        r = 0: complete disorder
        r = 1: perfect synchronization
        
        Returns:
            Order parameter [0, 1]
        """
        if not self.phase_history:
            return 0.0
        
        # Get current phases
        current_phases = [
            phases[-1] for phases in self.phase_history.values()
            if len(phases) > 0
        ]
        
        if not current_phases:
            return 0.0
        
        # Compute complex average
        avg_complex = np.mean([
            complex(math.cos(phase), math.sin(phase))
            for phase in current_phases
        ])
        
        return abs(avg_complex)
    
    def detect_synchronization(self) -> Optional[EmergentPattern]:
        """
        Detect synchronized behavior.
        
        Returns:
            EmergentPattern if synchronization detected, None otherwise
        """
        order = self.compute_order_parameter()
        
        if order > self.coupling_threshold:
            return EmergentPattern(
                pattern_id=f"sync_{len(self.phase_history)}",
                pattern_type="synchronization",
                strength=order,
                components=list(self.phase_history.keys()),
                emergence_time=0,
                description=f"Synchronized oscillation (r={order:.3f})"
            )
        
        return None


class CriticalityDetector:
    """
    Detect criticality and phase transitions.
    
    Systems at criticality show power-law distributions,
    long-range correlations, and scale-free behavior.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.event_sizes: deque = deque(maxlen=window_size)
        
    def add_event(self, event_size: float):
        """Add event size to distribution."""
        self.event_sizes.append(event_size)
    
    def fit_power_law(self) -> Tuple[float, float]:
        """
        Fit power-law distribution: P(s) ~ s^(-α)
        
        Returns:
            (exponent α, R² goodness of fit)
        """
        if len(self.event_sizes) < 10:
            return 0.0, 0.0
        
        sizes = np.array(list(self.event_sizes))
        sizes = sizes[sizes > 0]  # Remove zeros
        
        if len(sizes) < 10:
            return 0.0, 0.0
        
        # Log-log linear regression
        log_sizes = np.log(sizes)
        log_probs = np.log(np.arange(1, len(sizes) + 1) / len(sizes))
        
        # Simple linear regression in log-log space
        coeffs = np.polyfit(log_sizes, log_probs, 1)
        alpha = -coeffs[0]  # Power-law exponent
        
        # Compute R²
        predicted = coeffs[0] * log_sizes + coeffs[1]
        ss_res = np.sum((log_probs - predicted) ** 2)
        ss_tot = np.sum((log_probs - np.mean(log_probs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return alpha, r_squared
    
    def detect_criticality(self) -> Optional[EmergentPattern]:
        """
        Detect critical state.
        
        Critical systems have α ≈ 1.5-2.5 (depends on dimension)
        and high R² fit to power law.
        """
        alpha, r2 = self.fit_power_law()
        
        # Check for criticality indicators
        is_power_law = r2 > 0.8  # Good power-law fit
        is_critical_exponent = 1.2 < alpha < 3.0
        
        if is_power_law and is_critical_exponent:
            return EmergentPattern(
                pattern_id="criticality",
                pattern_type="criticality",
                strength=r2,
                components=["system"],
                emergence_time=0,
                description=f"Critical state detected (α={alpha:.2f}, R²={r2:.3f})"
            )
        
        return None


class SelfOrganizationDetector:
    """
    Detect self-organization and spontaneous pattern formation.
    
    Based on entropy reduction and order parameter increases.
    """
    
    def __init__(self):
        self.state_history: List[Dict] = []
        
    def add_state(self, state: Dict[str, Any]):
        """Add system state snapshot."""
        self.state_history.append(state.copy())
        
        # Keep last 1000 states
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
    
    def compute_entropy(self, state: Dict[str, Any]) -> float:
        """
        Compute Shannon entropy of state distribution.
        
        H = -Σ p(x) log₂ p(x)
        """
        # Convert state to discrete distribution
        values = list(state.values())
        
        if not values:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(values, bins=10, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize to probabilities
        
        # Compute entropy
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def detect_self_organization(self) -> Optional[EmergentPattern]:
        """
        Detect self-organization via entropy reduction.
        
        Self-organization shows decreasing entropy over time
        while maintaining or increasing order.
        """
        if len(self.state_history) < 10:
            return None
        
        # Compute entropy trend
        recent_entropies = [
            self.compute_entropy(state)
            for state in self.state_history[-10:]
        ]
        
        # Check for decreasing entropy (order increase)
        entropy_trend = np.polyfit(range(len(recent_entropies)), recent_entropies, 1)[0]
        
        if entropy_trend < -0.1:  # Significant entropy reduction
            entropy_reduction = abs(entropy_trend)
            
            return EmergentPattern(
                pattern_id="self_org",
                pattern_type="self_organization",
                strength=min(entropy_reduction * 10, 1.0),
                components=list(self.state_history[-1].keys()),
                emergence_time=len(self.state_history) - 10,
                description=f"Self-organization detected (ΔH={entropy_trend:.3f})"
            )
        
        return None


class CascadeDetector:
    """
    Detect cascade and avalanche dynamics.
    
    Avalanches are characteristic of self-organized criticality (SOC).
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.events: List[Tuple[int, float]] = []  # (time, magnitude)
        self.avalanche_in_progress = False
        self.current_avalanche_size = 0
        self.avalanche_sizes: List[int] = []
        
    def add_event(self, time: int, magnitude: float):
        """
        Add event to detect cascades.
        
        Args:
            time: Event timestamp
            magnitude: Event magnitude
        """
        self.events.append((time, magnitude))
        
        if magnitude > self.threshold:
            if not self.avalanche_in_progress:
                self.avalanche_in_progress = True
                self.current_avalanche_size = 1
            else:
                self.current_avalanche_size += 1
        else:
            if self.avalanche_in_progress:
                self.avalanche_sizes.append(self.current_avalanche_size)
                self.avalanche_in_progress = False
                self.current_avalanche_size = 0
    
    def detect_cascade(self) -> Optional[EmergentPattern]:
        """
        Detect cascading behavior.
        
        Returns pattern if recent large avalanche detected.
        """
        if not self.avalanche_sizes:
            return None
        
        recent_avalanches = self.avalanche_sizes[-10:]
        max_size = max(recent_avalanches) if recent_avalanches else 0
        mean_size = np.mean(recent_avalanches) if recent_avalanches else 0
        
        # Large avalanche = 3× mean
        if max_size > 3 * mean_size and max_size > 5:
            return EmergentPattern(
                pattern_id=f"cascade_{len(self.avalanche_sizes)}",
                pattern_type="cascade",
                strength=min(max_size / (3 * mean_size), 1.0),
                components=["system"],
                emergence_time=len(self.events) - max_size,
                description=f"Cascade detected (size={max_size}, mean={mean_size:.1f})"
            )
        
        return None


class EmergentBehaviorMonitor:
    """
    Unified monitor for emergent behavior detection.
    
    Integrates multiple detectors for comprehensive emergence tracking.
    """
    
    def __init__(self):
        self.sync_detector = SynchronizationDetector()
        self.criticality_detector = CriticalityDetector()
        self.self_org_detector = SelfOrganizationDetector()
        self.cascade_detector = CascadeDetector()
        self.detected_patterns: List[EmergentPattern] = []
        
    def update(self, system_state: Dict) -> List[EmergentPattern]:
        """
        Update all detectors and check for emergent patterns.
        
        Args:
            system_state: Current system state
            
        Returns:
            List of detected emergent patterns
        """
        patterns = []
        
        # Update detectors
        self.self_org_detector.add_state(system_state)
        
        # Check for patterns
        if pattern := self.sync_detector.detect_synchronization():
            patterns.append(pattern)
            
        if pattern := self.criticality_detector.detect_criticality():
            patterns.append(pattern)
            
        if pattern := self.self_org_detector.detect_self_organization():
            patterns.append(pattern)
            
        if pattern := self.cascade_detector.detect_cascade():
            patterns.append(pattern)
        
        # Store detected patterns
        self.detected_patterns.extend(patterns)
        
        return patterns
    
    def get_emergence_summary(self) -> Dict:
        """Get summary of detected emergence."""
        pattern_counts = defaultdict(int)
        for pattern in self.detected_patterns:
            pattern_counts[pattern.pattern_type] += 1
        
        return {
            'total_patterns': len(self.detected_patterns),
            'by_type': dict(pattern_counts),
            'recent': self.detected_patterns[-10:] if self.detected_patterns else []
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("EMERGENT BEHAVIOR DETECTION - DEMONSTRATION")
    print("=" * 80)
    
    # 1. Synchronization Detection
    print("\n1. SYNCHRONIZATION DETECTION (Kuramoto Model)")
    print("-" * 80)
    
    sync_det = SynchronizationDetector()
    
    # Simulate coupled oscillators converging to sync
    np.random.seed(42)
    phases = np.random.uniform(0, 2 * math.pi, 10)
    
    for step in range(50):
        # Coupling dynamics (simplified)
        mean_phase = np.mean(phases)
        coupling_strength = 0.1
        phases += 0.1 + coupling_strength * np.sin(mean_phase - phases)
        
        for i, phase in enumerate(phases):
            sync_det.add_oscillator_state(f"osc_{i}", phase)
    
    order_param = sync_det.compute_order_parameter()
    sync_pattern = sync_det.detect_synchronization()
    
    print(f"Number of oscillators: {len(phases)}")
    print(f"Order parameter r: {order_param:.4f}")
    print(f"Synchronization detected: {sync_pattern is not None}")
    if sync_pattern:
        print(f"Pattern: {sync_pattern.description}")
    
    # 2. Criticality Detection
    print("\n2. CRITICALITY DETECTION (Power-Law Distribution)")
    print("-" * 80)
    
    crit_det = CriticalityDetector()
    
    # Generate power-law distributed events
    for _ in range(500):
        # Power-law: s^(-2)
        event_size = np.random.pareto(1.0) + 1  # α = 2
        crit_det.add_event(event_size)
    
    alpha, r2 = crit_det.fit_power_law()
    crit_pattern = crit_det.detect_criticality()
    
    print(f"Power-law exponent α: {alpha:.3f}")
    print(f"R² goodness of fit: {r2:.3f}")
    print(f"Criticality detected: {crit_pattern is not None}")
    if crit_pattern:
        print(f"Pattern: {crit_pattern.description}")
    
    # 3. Self-Organization Detection
    print("\n3. SELF-ORGANIZATION DETECTION (Entropy Reduction)")
    print("-" * 80)
    
    self_org_det = SelfOrganizationDetector()
    
    # Simulate system moving from disorder to order
    for t in range(20):
        # Start random, gradually become ordered
        order_param_local = t / 20.0
        state = {
            f"var_{i}": np.random.beta(1 + order_param_local * 10, 1 + (1 - order_param_local) * 10)
            for i in range(5)
        }
        self_org_det.add_state(state)
    
    self_org_pattern = self_org_det.detect_self_organization()
    
    entropies = [self_org_det.compute_entropy(s) for s in self_org_det.state_history]
    print(f"Initial entropy: {entropies[0]:.3f}")
    print(f"Final entropy: {entropies[-1]:.3f}")
    print(f"Entropy reduction: {entropies[0] - entropies[-1]:.3f}")
    print(f"Self-organization detected: {self_org_pattern is not None}")
    if self_org_pattern:
        print(f"Pattern: {self_org_pattern.description}")
    
    # 4. Cascade Detection
    print("\n4. CASCADE DETECTION (Avalanche Dynamics)")
    print("-" * 80)
    
    cascade_det = CascadeDetector(threshold=0.5)
    
    # Simulate avalanches (SOC-like)
    for t in range(100):
        if np.random.random() < 0.1:  # 10% chance of avalanche
            avalanche_size = int(np.random.pareto(1.5) + 1)
            for _ in range(avalanche_size):
                cascade_det.add_event(t, np.random.uniform(0.5, 1.0))
        else:
            cascade_det.add_event(t, np.random.uniform(0, 0.3))
    
    cascade_pattern = cascade_det.detect_cascade()
    
    print(f"Total avalanches: {len(cascade_det.avalanche_sizes)}")
    if cascade_det.avalanche_sizes:
        print(f"Mean avalanche size: {np.mean(cascade_det.avalanche_sizes):.1f}")
        print(f"Max avalanche size: {max(cascade_det.avalanche_sizes)}")
    print(f"Cascade detected: {cascade_pattern is not None}")
    if cascade_pattern:
        print(f"Pattern: {cascade_pattern.description}")
    
    # 5. Unified Monitor
    print("\n5. UNIFIED EMERGENT BEHAVIOR MONITOR")
    print("-" * 80)
    
    monitor = EmergentBehaviorMonitor()
    
    # Simulate system evolution
    for t in range(30):
        state = {f"component_{i}": np.random.random() for i in range(5)}
        
        # Add events for detectors
        monitor.criticality_detector.add_event(np.random.pareto(1.5) + 1)
        monitor.cascade_detector.add_event(t, np.random.random())
        
        for i in range(5):
            phase = t * 0.1 + i * 0.2
            monitor.sync_detector.add_oscillator_state(f"comp_{i}", phase)
        
        patterns = monitor.update(state)
        
        if patterns:
            print(f"  t={t}: {len(patterns)} pattern(s) detected")
    
    summary = monitor.get_emergence_summary()
    print(f"\nEmergence Summary:")
    print(f"  Total patterns detected: {summary['total_patterns']}")
    print(f"  By type: {summary['by_type']}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Achievements:")
    print("✓ Synchronization detection via Kuramoto model")
    print("✓ Criticality detection via power-law fitting")
    print("✓ Self-organization via entropy dynamics")
    print("✓ Cascade detection for avalanche phenomena")
    print("✓ Unified emergent behavior monitoring")
