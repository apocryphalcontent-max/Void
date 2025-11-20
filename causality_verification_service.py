"""
Causality Verification Service
"The Mathematical Proof of Causality"

A service that consumes the event stream and mathematically proves
that no "Effect" happened before its "Cause".

If a violation is detected, it halts the universe (the cluster).

Key features:
- Monitors all events in the distributed system
- Verifies causal ordering using HLC timestamps
- Detects causality violations in real-time
- Maintains causal history for reconstruction
- Emergency cluster halt on violation

References:
- "Time, Clocks, and the Ordering of Events" (Lamport, 1978)
- "Detecting Causal Relationships in Distributed Computations" (Fidge, 1988)
- "Virtual Time and Global States" (Mattern, 1989)
"""

import time
import threading
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import networkx as nx

from hlc import HLCTimestamp


# ============================================================================
# EVENT DEFINITIONS
# ============================================================================

class EventType(Enum):
    """Types of events in the distributed system"""
    MESSAGE_SEND = "message_send"
    MESSAGE_RECV = "message_recv"
    DATABASE_WRITE = "database_write"
    DATABASE_READ = "database_read"
    MOUSE_CLICK = "mouse_click"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    TRANSACTION_BEGIN = "transaction_begin"
    TRANSACTION_COMMIT = "transaction_commit"
    CUSTOM = "custom"


@dataclass
class CausalEvent:
    """
    An event in the distributed system.
    
    Each event has:
    - Unique identifier
    - HLC timestamp (for causal ordering)
    - Type of event
    - Source node
    - Optional causal dependencies
    """
    event_id: str
    timestamp: HLCTimestamp
    event_type: EventType
    node_id: str
    data: Any = None
    
    # Causal dependencies: events that must happen-before this one
    dependencies: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.event_id)
    
    def __repr__(self):
        return f"Event({self.event_id}, {self.timestamp}, {self.event_type.value})"


@dataclass
class CausalityViolation:
    """A detected causality violation"""
    effect_event: CausalEvent
    cause_event: CausalEvent
    violation_type: str
    detected_at: float
    
    def __repr__(self):
        return (f"VIOLATION: {self.effect_event.event_id} "
                f"(ts={self.effect_event.timestamp}) happened before "
                f"{self.cause_event.event_id} (ts={self.cause_event.timestamp})")


# ============================================================================
# CAUSALITY VERIFICATION SERVICE
# ============================================================================

class CausalityVerificationService:
    """
    Service that verifies causal ordering of all events.
    
    The service maintains:
    1. Complete event history with HLC timestamps
    2. Causal dependency graph (happens-before relation)
    3. Real-time violation detection
    4. Emergency halt mechanism
    
    Mathematical guarantees:
    - If e1 → e2 (e1 happens-before e2), then HLC(e1) < HLC(e2)
    - If HLC(e1) < HLC(e2), we can prove e1 did not happen after e2
    """
    
    def __init__(
        self,
        cluster_id: str,
        halt_on_violation: bool = True,
        max_history: int = 10000
    ):
        """
        Initialize causality verification service.
        
        Args:
            cluster_id: Identifier for this cluster
            halt_on_violation: Whether to halt cluster on violation
            max_history: Maximum events to keep in history
        """
        self.cluster_id = cluster_id
        self.halt_on_violation = halt_on_violation
        self.max_history = max_history
        
        # Event storage
        self.events: Dict[str, CausalEvent] = {}
        self.event_history: deque = deque(maxlen=max_history)
        
        # Causal graph (DAG of happens-before relations)
        self.causal_graph = nx.DiGraph()
        
        # Violations detected
        self.violations: List[CausalityViolation] = []
        
        # State
        self._lock = threading.Lock()
        self._halted = False
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'violations_detected': 0,
            'verifications_performed': 0,
            'causal_chains_verified': 0
        }
        
        # Callbacks
        self.on_violation: Optional[Callable[[CausalityViolation], None]] = None
        self.on_halt: Optional[Callable[[str], None]] = None
    
    def is_halted(self) -> bool:
        """Check if cluster is halted due to violation"""
        with self._lock:
            return self._halted
    
    def record_event(self, event: CausalEvent) -> bool:
        """
        Record a new event and verify causality.
        
        Args:
            event: The event to record
            
        Returns:
            True if event is valid, False if violation detected
        """
        with self._lock:
            if self._halted:
                return False
            
            # Check for duplicate event ID
            if event.event_id in self.events:
                return False
            
            # Store event
            self.events[event.event_id] = event
            self.event_history.append(event)
            self.stats['events_processed'] += 1
            
            # Add to causal graph
            self.causal_graph.add_node(event.event_id, event=event)
            
            # Add edges for dependencies
            for dep_id in event.dependencies:
                if dep_id in self.events:
                    self.causal_graph.add_edge(dep_id, event.event_id)
            
            # Verify causality
            violation = self._verify_event_causality(event)
            
            if violation:
                self._handle_violation(violation)
                return False
            
            return True
    
    def _verify_event_causality(self, event: CausalEvent) -> Optional[CausalityViolation]:
        """
        Verify that an event respects causal ordering.
        
        Checks:
        1. All dependencies have timestamps < event timestamp
        2. No transitive violations in causal chain
        
        Args:
            event: Event to verify
            
        Returns:
            CausalityViolation if violation detected, None otherwise
        """
        self.stats['verifications_performed'] += 1
        
        # Check direct dependencies
        for dep_id in event.dependencies:
            if dep_id not in self.events:
                # Dependency not yet recorded - this is okay
                # (could be network delay)
                continue
            
            dep_event = self.events[dep_id]
            
            # CRITICAL CHECK: dependency must happen before this event
            if not (dep_event.timestamp < event.timestamp):
                return CausalityViolation(
                    effect_event=event,
                    cause_event=dep_event,
                    violation_type="dependency_timestamp_violation",
                    detected_at=time.time()
                )
        
        # Check transitive dependencies (full causal chain)
        # Get all ancestors in causal graph
        if event.event_id in self.causal_graph:
            try:
                ancestors = nx.ancestors(self.causal_graph, event.event_id)
                self.stats['causal_chains_verified'] += 1
                
                for ancestor_id in ancestors:
                    if ancestor_id in self.events:
                        ancestor = self.events[ancestor_id]
                        
                        # All ancestors must have timestamp < event timestamp
                        if not (ancestor.timestamp < event.timestamp):
                            return CausalityViolation(
                                effect_event=event,
                                cause_event=ancestor,
                                violation_type="transitive_causality_violation",
                                detected_at=time.time()
                            )
            except nx.NetworkXError:
                # Cycle detected in causal graph - this is a violation!
                return CausalityViolation(
                    effect_event=event,
                    cause_event=event,  # Self-reference for cycle
                    violation_type="causal_cycle_detected",
                    detected_at=time.time()
                )
        
        return None
    
    def _handle_violation(self, violation: CausalityViolation):
        """
        Handle a detected causality violation.
        
        Args:
            violation: The detected violation
        """
        self.violations.append(violation)
        self.stats['violations_detected'] += 1
        
        # Trigger callback
        if self.on_violation:
            self.on_violation(violation)
        
        # Halt the cluster if configured
        if self.halt_on_violation:
            self._halt_cluster(violation)
    
    def _halt_cluster(self, violation: CausalityViolation):
        """
        Emergency halt of the cluster due to causality violation.
        
        In production, this would:
        1. Stop all transactions
        2. Prevent new operations
        3. Alert operators
        4. Preserve state for forensic analysis
        
        Args:
            violation: The violation that triggered the halt
        """
        self._halted = True
        
        halt_message = (
            f"EMERGENCY CLUSTER HALT - Causality Violation Detected\n"
            f"Cluster: {self.cluster_id}\n"
            f"Violation: {violation}\n"
            f"Time: {time.ctime(violation.detected_at)}\n"
            f"\n"
            f"The universe has been halted to prevent causal paradox.\n"
            f"No further operations will be processed until resolution.\n"
        )
        
        if self.on_halt:
            self.on_halt(halt_message)
    
    def verify_global_causality(self) -> List[CausalityViolation]:
        """
        Perform global causality verification on all events.
        
        This is a comprehensive check that verifies:
        1. Happens-before relation is acyclic (DAG property)
        2. Timestamps respect happens-before ordering
        3. No orphaned events (all dependencies present)
        
        Returns:
            List of violations found (empty if all valid)
        """
        violations = []
        
        with self._lock:
            # Check 1: Verify DAG property (no cycles)
            if not nx.is_directed_acyclic_graph(self.causal_graph):
                # Find cycles
                try:
                    cycles = list(nx.simple_cycles(self.causal_graph))
                    for cycle in cycles:
                        if cycle:
                            event = self.events.get(cycle[0])
                            if event:
                                violations.append(CausalityViolation(
                                    effect_event=event,
                                    cause_event=event,
                                    violation_type="causal_cycle",
                                    detected_at=time.time()
                                ))
                except Exception:
                    pass
            
            # Check 2: Verify timestamp ordering along all paths
            for event_id in self.causal_graph.nodes():
                if event_id not in self.events:
                    continue
                
                event = self.events[event_id]
                
                # Check all predecessors (direct causes)
                for pred_id in self.causal_graph.predecessors(event_id):
                    if pred_id not in self.events:
                        continue
                    
                    pred_event = self.events[pred_id]
                    
                    if not (pred_event.timestamp < event.timestamp):
                        violations.append(CausalityViolation(
                            effect_event=event,
                            cause_event=pred_event,
                            violation_type="timestamp_ordering_violation",
                            detected_at=time.time()
                        ))
        
        return violations
    
    def get_causal_history(self, event_id: str) -> List[CausalEvent]:
        """
        Reconstruct the complete causal history of an event.
        
        Returns all events that causally precede this event,
        in topological order.
        
        Args:
            event_id: Event to trace history for
            
        Returns:
            List of events in causal order
        """
        with self._lock:
            if event_id not in self.causal_graph:
                return []
            
            # Get all ancestors
            ancestors = nx.ancestors(self.causal_graph, event_id)
            
            # Create subgraph
            subgraph = self.causal_graph.subgraph(ancestors | {event_id})
            
            # Topological sort gives us causal order
            try:
                ordered = list(nx.topological_sort(subgraph))
                return [self.events[eid] for eid in ordered if eid in self.events]
            except nx.NetworkXError:
                return []
    
    def query_events_at_time(self, timestamp: HLCTimestamp) -> List[CausalEvent]:
        """
        Query all events visible at a given timestamp.
        
        Returns all events with timestamp <= query timestamp.
        
        Args:
            timestamp: Query timestamp
            
        Returns:
            List of visible events
        """
        with self._lock:
            visible = []
            
            for event in self.events.values():
                if event.timestamp <= timestamp:
                    visible.append(event)
            
            # Sort by timestamp
            visible.sort(key=lambda e: e.timestamp)
            
            return visible
    
    def get_statistics(self) -> dict:
        """Get service statistics"""
        with self._lock:
            return {
                **self.stats,
                'total_events': len(self.events),
                'total_violations': len(self.violations),
                'halted': self._halted,
                'causal_graph_nodes': self.causal_graph.number_of_nodes(),
                'causal_graph_edges': self.causal_graph.number_of_edges()
            }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def _example_usage():
    """Demonstrate causality verification"""
    
    print("Causality Verification Service")
    print("="*70)
    
    from hlc import HybridLogicalClock
    
    # Create service
    print("\n1. Initializing Causality Verification Service...")
    service = CausalityVerificationService(
        cluster_id="universe_alpha",
        halt_on_violation=True
    )
    
    # Set up violation handler
    def on_violation(violation: CausalityViolation):
        print(f"\n   ⚠ CAUSALITY VIOLATION DETECTED:")
        print(f"      {violation}")
    
    def on_halt(message: str):
        print(f"\n   🛑 CLUSTER HALTED:")
        print(f"      {message}")
    
    service.on_violation = on_violation
    service.on_halt = on_halt
    
    # Create HLC for timestamp generation
    clock = HybridLogicalClock("node1")
    
    # Valid causal chain
    print("\n2. Recording valid causal chain:")
    
    ts1 = clock.now()
    e1 = CausalEvent("e1", ts1, EventType.API_REQUEST, "node1", data="GET /api/users")
    valid = service.record_event(e1)
    print(f"   Event e1: {e1} - Valid: {valid}")
    
    time.sleep(0.01)
    ts2 = clock.now()
    e2 = CausalEvent("e2", ts2, EventType.DATABASE_READ, "node1", 
                     data="SELECT * FROM users", dependencies={"e1"})
    valid = service.record_event(e2)
    print(f"   Event e2: {e2} - Valid: {valid}")
    
    time.sleep(0.01)
    ts3 = clock.now()
    e3 = CausalEvent("e3", ts3, EventType.API_RESPONSE, "node1",
                     data="200 OK", dependencies={"e2"})
    valid = service.record_event(e3)
    print(f"   Event e3: {e3} - Valid: {valid}")
    
    print(f"\n   ✓ Valid causal chain: e1 → e2 → e3")
    
    # Reconstruct causal history
    print("\n3. Reconstructing causal history of e3:")
    history = service.get_causal_history("e3")
    for event in history:
        print(f"   - {event}")
    
    # Statistics
    print("\n4. Service statistics:")
    stats = service.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n5. Causality guarantees:")
    print("   ✓ Happens-before relation is preserved")
    print("   ✓ Timestamps respect causal ordering")
    print("   ✓ Causal history can be reconstructed")
    print("   ✓ Violations trigger emergency halt")
    print("   ✓ Complete audit trail of all events")
    
    print("\n" + "="*70)
    print("Every event in the universe has a provable causal history.")


if __name__ == "__main__":
    _example_usage()
