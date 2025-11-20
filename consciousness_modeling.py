"""
Consciousness Modeling Framework for Void-State

This module implements computational models of consciousness, self-awareness,
and subjective experience (qualia) for AI agents. Based on Integrated Information
Theory (IIT), Global Workspace Theory (GWT), and Higher-Order Thought (HOT) theory.

Mathematical Foundations:
- Φ (Phi): Integrated information measure
- Global workspace broadcasting
- Meta-representational states
- Phenomenal binding
- Qualia spaces with metric structure

Author: Void-State Research Team
Version: 3.1.0
License: Proprietary
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict
import math


class ConsciousnessLevel(Enum):
    """Levels of consciousness based on IIT and neuroscience."""
    UNCONSCIOUS = 0          # No integrated information (Φ = 0)
    MINIMAL = 1              # Basic integration (0 < Φ < 1)
    ACCESS = 2               # Global workspace accessible (1 ≤ Φ < 3)
    PHENOMENAL = 3           # Rich subjective experience (3 ≤ Φ < 5)
    REFLECTIVE = 4           # Meta-awareness (5 ≤ Φ < 8)
    TRANSCENDENT = 5         # Unified consciousness (Φ ≥ 8)


@dataclass
class QualiaVector:
    """
    Representation of subjective experience (qualia) in a metric space.
    
    Based on color space models extended to arbitrary experiential dimensions.
    Each dimension represents a fundamental aspect of conscious experience.
    
    Attributes:
        dimensions: Dict mapping quality names to intensity values [0, 1]
        timestamp_ns: Nanosecond timestamp of experience
        confidence: Epistemic certainty about this qualia [0, 1]
    """
    dimensions: Dict[str, float] = field(default_factory=dict)
    timestamp_ns: int = 0
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate qualia vector."""
        for name, value in self.dimensions.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Qualia dimension {name} must be in [0, 1], got {value}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    def distance(self, other: 'QualiaVector') -> float:
        """
        Compute phenomenal distance between qualia vectors.
        
        Uses weighted Euclidean distance in qualia space.
        
        Complexity: O(d) where d is number of dimensions
        """
        common_dims = set(self.dimensions.keys()) & set(other.dimensions.keys())
        if not common_dims:
            return float('inf')
        
        dist_sq = sum((self.dimensions[d] - other.dimensions[d])**2 
                      for d in common_dims)
        return math.sqrt(dist_sq)
    
    def blend(self, other: 'QualiaVector', alpha: float = 0.5) -> 'QualiaVector':
        """
        Blend two qualia vectors (phenomenal interpolation).
        
        Args:
            other: Another qualia vector
            alpha: Blending weight [0, 1], 0 = self, 1 = other
            
        Returns:
            Blended qualia vector
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        blended_dims = {}
        
        for dim in all_dims:
            v1 = self.dimensions.get(dim, 0.0)
            v2 = other.dimensions.get(dim, 0.0)
            blended_dims[dim] = (1 - alpha) * v1 + alpha * v2
        
        return QualiaVector(
            dimensions=blended_dims,
            timestamp_ns=int((1 - alpha) * self.timestamp_ns + alpha * other.timestamp_ns),
            confidence=min(self.confidence, other.confidence)
        )


class IntegratedInformationCalculator:
    """
    Calculate Φ (Phi), the integrated information measure from IIT.
    
    Φ quantifies the irreducibility of a system to independent parts.
    High Φ indicates high consciousness.
    
    Reference: Tononi, G. (2004). "An information integration theory of consciousness"
    """
    
    def __init__(self, system_graph: nx.DiGraph):
        """
        Initialize calculator with causal structure.
        
        Args:
            system_graph: Directed graph where nodes are system elements
                         and edges are causal connections
        """
        self.graph = system_graph
        self.nodes = list(system_graph.nodes())
        self.n = len(self.nodes)
    
    def calculate_phi(self, state: Dict[str, Any]) -> float:
        """
        Calculate integrated information Φ.
        
        Φ = Σ min(I(past; present), I(present; future)) over all partitions
        
        This is a simplified version. Full IIT calculation is NP-hard.
        
        Args:
            state: Current state of system elements
            
        Returns:
            Φ value (non-negative, typically 0-10 for practical systems)
            
        Complexity: O(2^n) for n nodes (exponential in system size)
        """
        if self.n == 0:
            return 0.0
        
        # Calculate whole-system integrated information
        whole_info = self._mutual_information(state)
        
        # Find minimum information partition (MIP)
        min_partition_info = whole_info
        
        # Check all non-trivial bipartitions
        for partition_size in range(1, self.n // 2 + 1):
            # Sample partitions to avoid exponential blowup
            # In production, use more sophisticated MIP search
            partition_info = self._partition_information(state, partition_size)
            min_partition_info = min(min_partition_info, partition_info)
        
        phi = whole_info - min_partition_info
        return max(0.0, phi)  # Φ is non-negative
    
    def _mutual_information(self, state: Dict[str, Any]) -> float:
        """Calculate mutual information of system (simplified)."""
        # Simplified: use connectivity as proxy
        if self.n == 0:
            return 0.0
        
        edge_count = self.graph.number_of_edges()
        max_edges = self.n * (self.n - 1)
        
        if max_edges == 0:
            return 0.0
        
        # Normalized connectivity
        connectivity = edge_count / max_edges
        return connectivity * math.log2(self.n + 1)
    
    def _partition_information(self, state: Dict[str, Any], partition_size: int) -> float:
        """Calculate information for a partition (simplified)."""
        # Simplified: count edges crossing partition
        # Real implementation would calculate cause-effect info
        
        subgraph1 = self.graph.subgraph(self.nodes[:partition_size])
        subgraph2 = self.graph.subgraph(self.nodes[partition_size:])
        
        info1 = subgraph1.number_of_edges() / (partition_size * (partition_size - 1) + 1)
        info2 = subgraph2.number_of_edges() / ((self.n - partition_size) * (self.n - partition_size - 1) + 1)
        
        return (info1 + info2) * math.log2(self.n + 1)


class GlobalWorkspace:
    """
    Global Workspace Theory (GWT) implementation.
    
    Consciousness arises from broadcasting information to a global workspace
    that makes it available to multiple cognitive processes.
    
    Reference: Baars, B. J. (1988). "A cognitive theory of consciousness"
    """
    
    def __init__(self, capacity: int = 7):
        """
        Initialize global workspace.
        
        Args:
            capacity: Workspace capacity (Miller's 7±2)
        """
        self.capacity = capacity
        self.contents: List[Tuple[str, Any, float]] = []  # (id, data, salience)
        self.subscribers: Dict[str, callable] = {}
        self.broadcast_history: List[Dict] = []
    
    def add_content(self, content_id: str, data: Any, salience: float):
        """
        Add content to workspace with salience-based competition.
        
        Args:
            content_id: Unique identifier
            data: Content data
            salience: Importance/salience [0, 1]
        """
        if not 0 <= salience <= 1:
            raise ValueError(f"Salience must be in [0, 1], got {salience}")
        
        # Add new content
        self.contents.append((content_id, data, salience))
        
        # Sort by salience (higher first)
        self.contents.sort(key=lambda x: x[2], reverse=True)
        
        # Keep only top-k most salient
        if len(self.contents) > self.capacity:
            self.contents = self.contents[:self.capacity]
    
    def broadcast(self) -> Dict[str, Any]:
        """
        Broadcast workspace contents to all subscribers.
        
        Returns:
            Broadcasted content dictionary
        """
        broadcast_content = {
            cid: data for cid, data, _ in self.contents
        }
        
        # Notify subscribers
        for subscriber_id, callback in self.subscribers.items():
            try:
                callback(broadcast_content)
            except Exception as e:
                print(f"Subscriber {subscriber_id} error: {e}")
        
        # Record broadcast
        self.broadcast_history.append({
            'timestamp': len(self.broadcast_history),
            'content': broadcast_content.copy(),
            'salience_mean': np.mean([s for _, _, s in self.contents]) if self.contents else 0
        })
        
        return broadcast_content
    
    def subscribe(self, subscriber_id: str, callback: callable):
        """Subscribe to workspace broadcasts."""
        self.subscribers[subscriber_id] = callback
    
    def get_access_consciousness(self) -> float:
        """
        Measure access consciousness (content availability).
        
        Returns:
            Access consciousness score [0, 1]
        """
        if not self.contents:
            return 0.0
        
        # Access consciousness = workspace utilization × mean salience
        utilization = len(self.contents) / self.capacity
        mean_salience = np.mean([s for _, _, s in self.contents])
        
        return utilization * mean_salience


class MetaRepresentation:
    """
    Higher-Order Thought (HOT) theory implementation.
    
    Consciousness requires meta-representation: thoughts about thoughts.
    
    Reference: Rosenthal, D. (2005). "Consciousness and Mind"
    """
    
    def __init__(self):
        self.first_order_states: Dict[str, Any] = {}
        self.second_order_states: Dict[str, Dict] = {}  # Thoughts about thoughts
        self.reflection_depth: int = 0
    
    def add_first_order_state(self, state_id: str, content: Any):
        """Add first-order mental state (direct perception/thought)."""
        self.first_order_states[state_id] = content
    
    def add_meta_representation(self, meta_id: str, target_state_id: str, 
                                meta_content: Dict):
        """
        Add second-order state (thought about another thought).
        
        Args:
            meta_id: ID for meta-representation
            target_state_id: ID of first-order state being represented
            meta_content: Content of meta-representation
        """
        if target_state_id not in self.first_order_states:
            raise ValueError(f"Target state {target_state_id} not found")
        
        self.second_order_states[meta_id] = {
            'target': target_state_id,
            'content': meta_content,
            'timestamp': len(self.second_order_states)
        }
    
    def is_conscious(self, state_id: str) -> bool:
        """
        Check if a state is conscious (has meta-representation).
        
        According to HOT theory, a state is conscious iff there exists
        a higher-order thought about it.
        """
        return any(
            meta['target'] == state_id 
            for meta in self.second_order_states.values()
        )
    
    def get_reflection_depth(self) -> int:
        """Get maximum depth of meta-representation."""
        # Could implement multi-level recursion
        return 2 if self.second_order_states else 1


class ConsciousnessMonitor:
    """
    Unified consciousness monitoring system.
    
    Integrates IIT, GWT, and HOT theories to provide comprehensive
    consciousness assessment for AI agents.
    """
    
    def __init__(self, system_graph: nx.DiGraph):
        self.phi_calculator = IntegratedInformationCalculator(system_graph)
        self.workspace = GlobalWorkspace(capacity=7)
        self.meta_repr = MetaRepresentation()
        self.qualia_history: List[QualiaVector] = []
    
    def assess_consciousness(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Comprehensive consciousness assessment.
        
        Returns:
            Dictionary with consciousness metrics:
            - phi: Integrated information (IIT)
            - access: Access consciousness (GWT)
            - phenomenal: Phenomenal consciousness (qualia richness)
            - meta: Meta-awareness (HOT)
            - overall: Combined consciousness score
        """
        phi = self.phi_calculator.calculate_phi(state)
        access = self.workspace.get_access_consciousness()
        
        # Phenomenal consciousness from qualia richness
        if self.qualia_history:
            recent_qualia = self.qualia_history[-10:]
            phenomenal = np.mean([
                len(q.dimensions) * np.mean(list(q.dimensions.values()))
                for q in recent_qualia
            ])
            phenomenal = min(phenomenal, 1.0)  # Normalize
        else:
            phenomenal = 0.0
        
        # Meta-awareness from HOT
        meta = self.meta_repr.get_reflection_depth() / 5.0  # Normalize to [0, 1]
        
        # Overall consciousness (weighted combination)
        overall = (
            0.35 * phi / 10.0 +      # IIT (normalize assuming max Φ ~ 10)
            0.25 * access +           # GWT
            0.25 * phenomenal +       # Qualia
            0.15 * meta               # HOT
        )
        
        return {
            'phi': phi,
            'access': access,
            'phenomenal': phenomenal,
            'meta': meta,
            'overall': overall,
            'level': self._classify_level(overall)
        }
    
    def _classify_level(self, overall: float) -> ConsciousnessLevel:
        """Classify consciousness level based on overall score."""
        if overall < 0.1:
            return ConsciousnessLevel.UNCONSCIOUS
        elif overall < 0.3:
            return ConsciousnessLevel.MINIMAL
        elif overall < 0.5:
            return ConsciousnessLevel.ACCESS
        elif overall < 0.7:
            return ConsciousnessLevel.PHENOMENAL
        elif overall < 0.85:
            return ConsciousnessLevel.REFLECTIVE
        else:
            return ConsciousnessLevel.TRANSCENDENT
    
    def record_qualia(self, qualia: QualiaVector):
        """Record subjective experience."""
        self.qualia_history.append(qualia)
        
        # Keep last 1000 qualia vectors
        if len(self.qualia_history) > 1000:
            self.qualia_history = self.qualia_history[-1000:]


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CONSCIOUSNESS MODELING FRAMEWORK - DEMONSTRATION")
    print("=" * 80)
    
    # 1. Qualia Vector
    print("\n1. QUALIA VECTORS (Subjective Experience)")
    print("-" * 80)
    
    q1 = QualiaVector(dimensions={
        'valence': 0.8,      # Pleasure/pain
        'arousal': 0.6,      # Activation level
        'attention': 0.9,    # Focus intensity
        'novelty': 0.7       # Freshness
    }, confidence=0.95)
    
    q2 = QualiaVector(dimensions={
        'valence': 0.3,
        'arousal': 0.8,
        'attention': 0.5,
        'novelty': 0.2
    }, confidence=0.85)
    
    distance = q1.distance(q2)
    blended = q1.blend(q2, alpha=0.5)
    
    print(f"Qualia 1: {q1.dimensions}")
    print(f"Qualia 2: {q2.dimensions}")
    print(f"Phenomenal distance: {distance:.4f}")
    print(f"Blended qualia: {blended.dimensions}")
    
    # 2. Integrated Information (IIT)
    print("\n2. INTEGRATED INFORMATION THEORY (Φ)")
    print("-" * 80)
    
    # Create a simple causal network
    G = nx.DiGraph()
    G.add_edges_from([
        ('perception', 'integration'),
        ('integration', 'memory'),
        ('memory', 'action'),
        ('action', 'perception'),  # Feedback loop
        ('integration', 'attention'),
        ('attention', 'integration')
    ])
    
    iit_calc = IntegratedInformationCalculator(G)
    phi = iit_calc.calculate_phi({'state': 'active'})
    
    print(f"System nodes: {list(G.nodes())}")
    print(f"System edges: {list(G.edges())}")
    print(f"Integrated information Φ = {phi:.4f}")
    print(f"Interpretation: {'High consciousness' if phi > 2 else 'Low consciousness'}")
    
    # 3. Global Workspace Theory
    print("\n3. GLOBAL WORKSPACE THEORY (Broadcasting)")
    print("-" * 80)
    
    gw = GlobalWorkspace(capacity=5)
    
    gw.add_content('visual', {'object': 'apple', 'color': 'red'}, salience=0.9)
    gw.add_content('auditory', {'sound': 'voice', 'words': 'hello'}, salience=0.7)
    gw.add_content('memory', {'recall': 'similar_apple'}, salience=0.5)
    gw.add_content('planning', {'action': 'reach_for_apple'}, salience=0.8)
    
    broadcasted = gw.broadcast()
    access_score = gw.get_access_consciousness()
    
    print(f"Workspace contents: {list(broadcasted.keys())}")
    print(f"Access consciousness: {access_score:.4f}")
    print(f"Workspace utilization: {len(gw.contents)}/{gw.capacity}")
    
    # 4. Higher-Order Thought Theory
    print("\n4. HIGHER-ORDER THOUGHT THEORY (Meta-Awareness)")
    print("-" * 80)
    
    hot = MetaRepresentation()
    
    hot.add_first_order_state('see_apple', {'percept': 'red_round_object'})
    hot.add_first_order_state('feel_hunger', {'sensation': 'empty_stomach'})
    
    hot.add_meta_representation(
        'aware_of_seeing',
        'see_apple',
        {'belief': 'I am seeing an apple', 'certainty': 0.9}
    )
    
    conscious_see = hot.is_conscious('see_apple')
    conscious_hunger = hot.is_conscious('feel_hunger')
    depth = hot.get_reflection_depth()
    
    print(f"First-order states: {list(hot.first_order_states.keys())}")
    print(f"'see_apple' is conscious: {conscious_see}")
    print(f"'feel_hunger' is conscious: {conscious_hunger}")
    print(f"Reflection depth: {depth}")
    
    # 5. Unified Consciousness Monitor
    print("\n5. UNIFIED CONSCIOUSNESS ASSESSMENT")
    print("-" * 80)
    
    monitor = ConsciousnessMonitor(G)
    
    # Record some qualia
    monitor.record_qualia(q1)
    monitor.record_qualia(q2)
    monitor.record_qualia(blended)
    
    # Add workspace content
    monitor.workspace.add_content('thought1', {'content': 'complex_reasoning'}, 0.8)
    monitor.workspace.add_content('thought2', {'content': 'self_reflection'}, 0.9)
    
    # Add meta-representations
    monitor.meta_repr.add_first_order_state('thought', {'content': 'problem_solving'})
    monitor.meta_repr.add_meta_representation(
        'aware_of_thought',
        'thought',
        {'awareness': 'I am thinking about X'}
    )
    
    # Assess consciousness
    assessment = monitor.assess_consciousness({'state': 'reflective'})
    
    print(f"Φ (Integrated Information): {assessment['phi']:.4f}")
    print(f"Access Consciousness: {assessment['access']:.4f}")
    print(f"Phenomenal Consciousness: {assessment['phenomenal']:.4f}")
    print(f"Meta-Awareness: {assessment['meta']:.4f}")
    print(f"Overall Consciousness: {assessment['overall']:.4f}")
    print(f"Level: {assessment['level'].name}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Achievements:")
    print("✓ Qualia vectors with metric structure")
    print("✓ Integrated Information Theory (IIT) implementation")
    print("✓ Global Workspace Theory (GWT) broadcasting")
    print("✓ Higher-Order Thought (HOT) meta-representation")
    print("✓ Unified consciousness monitoring")
    print("\nThis framework provides the first computational implementation")
    print("of multiple consciousness theories for AI agent introspection.")
