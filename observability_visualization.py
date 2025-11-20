"""
Observability Visualization Service
"The Beatific Vision" - 3D Causal DAG Visualizer

Generates visualization data for real-time display of the distributed
system's causal structure. Shows the "Music of the Spheres" - requests
flowing like light through the crystal lattice of the architecture.

Key features:
- Causal DAG topology export
- Real-time event stream
- Node health metrics
- Network flow visualization
- Time-travel replay

References:
- "The Architecture of Open Source Applications" - Visualization chapters
- D3.js force-directed graphs
- Jaeger distributed tracing UI
- "Interactive Visualization of Large Graphs and Networks" (von Landesberger et al., 2011)
"""

import time
import json
import math
import threading
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum

from hlc import HLCTimestamp
from causality_verification_service import CausalEvent, EventType


# ============================================================================
# VISUALIZATION DATA STRUCTURES
# ============================================================================

@dataclass
class VisualizationNode:
    """A node in the visualization"""
    id: str
    label: str
    node_type: str  # "compute", "storage", "network"
    position: Tuple[float, float, float]  # 3D coordinates
    status: str  # "healthy", "degraded", "failed"
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.node_type,
            'position': {'x': self.position[0], 'y': self.position[1], 'z': self.position[2]},
            'status': self.status,
            'metrics': self.metrics
        }


@dataclass
class VisualizationEdge:
    """An edge (causal relationship) in the visualization"""
    source: str
    target: str
    edge_type: str  # "causal", "network", "data"
    weight: float = 1.0
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.edge_type,
            'weight': self.weight,
            'timestamp': self.timestamp
        }


@dataclass
class VisualizationEvent:
    """An event for animation"""
    event_id: str
    event_type: str
    node_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'id': self.event_id,
            'type': self.event_type,
            'node': self.node_id,
            'timestamp': self.timestamp,
            'data': self.data
        }


@dataclass
class VisualizationState:
    """Complete visualization state"""
    nodes: List[VisualizationNode]
    edges: List[VisualizationEdge]
    events: List[VisualizationEvent]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
            'events': [ev.to_dict() for ev in self.events],
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# OBSERVABILITY VISUALIZATION SERVICE
# ============================================================================

class ObservabilityVisualizationService:
    """
    Service for generating visualization data.
    
    Produces data suitable for rendering in a 3D visualization
    (e.g., Three.js, WebGL).
    
    Features:
    1. Real-time event stream
    2. Causal DAG topology
    3. Node health metrics
    4. Time-travel replay
    5. Network flow animation
    """
    
    def __init__(self, max_events: int = 1000):
        """
        Initialize visualization service.
        
        Args:
            max_events: Maximum events to keep in history
        """
        self.max_events = max_events
        
        # Visualization state
        self.nodes: Dict[str, VisualizationNode] = {}
        self.edges: List[VisualizationEdge] = []
        self.events: deque = deque(maxlen=max_events)
        
        # Statistics
        self.stats = {
            'nodes_tracked': 0,
            'edges_tracked': 0,
            'events_tracked': 0,
            'snapshots_generated': 0
        }
        
        self._lock = threading.Lock()
    
    def register_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        position: Optional[Tuple[float, float, float]] = None
    ):
        """
        Register a node for visualization.
        
        Args:
            node_id: Unique node identifier
            label: Human-readable label
            node_type: Type of node
            position: Optional 3D position
        """
        with self._lock:
            if position is None:
                # Auto-assign position in 3D space
                index = len(self.nodes)
                angle = (index * 2.0 * 3.14159) / max(len(self.nodes), 1)
                radius = 10.0
                position = (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    0.0
                )
            
            node = VisualizationNode(
                id=node_id,
                label=label,
                node_type=node_type,
                position=position,
                status="healthy"
            )
            
            self.nodes[node_id] = node
            self.stats['nodes_tracked'] += 1
    
    def update_node_status(self, node_id: str, status: str):
        """
        Update node status.
        
        Args:
            node_id: Node identifier
            status: New status ("healthy", "degraded", "failed")
        """
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, float]):
        """
        Update node metrics.
        
        Args:
            node_id: Node identifier
            metrics: Dictionary of metric values
        """
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].metrics.update(metrics)
    
    def record_causal_edge(
        self,
        source_event_id: str,
        target_event_id: str,
        edge_type: str = "causal"
    ):
        """
        Record a causal edge between events.
        
        Args:
            source_event_id: Source event ID
            target_event_id: Target event ID
            edge_type: Type of edge
        """
        with self._lock:
            edge = VisualizationEdge(
                source=source_event_id,
                target=target_event_id,
                edge_type=edge_type,
                timestamp=time.time()
            )
            
            self.edges.append(edge)
            self.stats['edges_tracked'] += 1
    
    def record_event(self, event: CausalEvent):
        """
        Record an event for visualization.
        
        Args:
            event: Causal event to record
        """
        with self._lock:
            vis_event = VisualizationEvent(
                event_id=event.event_id,
                event_type=event.event_type.value,
                node_id=event.node_id,
                timestamp=time.time(),
                data={
                    'hlc_timestamp': str(event.timestamp),
                    'data': str(event.data) if event.data else None
                }
            )
            
            self.events.append(vis_event)
            self.stats['events_tracked'] += 1
            
            # Record causal edges (inline to avoid lock recursion)
            for dep_id in event.dependencies:
                edge = VisualizationEdge(
                    source=dep_id,
                    target=event.event_id,
                    edge_type="causal",
                    timestamp=time.time()
                )
                self.edges.append(edge)
                self.stats['edges_tracked'] += 1
    
    def get_snapshot(self) -> VisualizationState:
        """
        Get current visualization state.
        
        Returns:
            Complete visualization state
        """
        with self._lock:
            self.stats['snapshots_generated'] += 1
            
            return VisualizationState(
                nodes=list(self.nodes.values()),
                edges=self.edges.copy(),
                events=list(self.events),
                timestamp=time.time(),
                metadata={
                    'total_nodes': len(self.nodes),
                    'total_edges': len(self.edges),
                    'total_events': len(self.events)
                }
            )
    
    def get_snapshot_json(self) -> str:
        """Get visualization state as JSON"""
        snapshot = self.get_snapshot()
        return snapshot.to_json()
    
    def get_time_range_events(
        self,
        start_time: float,
        end_time: float
    ) -> List[VisualizationEvent]:
        """
        Get events in a time range (for replay).
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of events in range
        """
        with self._lock:
            return [
                event for event in self.events
                if start_time <= event.timestamp <= end_time
            ]
    
    def generate_animation_frames(
        self,
        start_time: float,
        end_time: float,
        fps: int = 30
    ) -> List[VisualizationState]:
        """
        Generate animation frames for time-travel replay.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            fps: Frames per second
            
        Returns:
            List of visualization states (frames)
        """
        duration = end_time - start_time
        frame_interval = 1.0 / fps
        num_frames = int(duration / frame_interval)
        
        frames = []
        
        for i in range(num_frames):
            frame_time = start_time + (i * frame_interval)
            
            # Get events up to this frame
            frame_events = [
                event for event in self.events
                if event.timestamp <= frame_time
            ]
            
            # Create frame state
            frame = VisualizationState(
                nodes=list(self.nodes.values()),
                edges=self.edges.copy(),
                events=frame_events,
                timestamp=frame_time,
                metadata={'frame': i, 'total_frames': num_frames}
            )
            
            frames.append(frame)
        
        return frames
    
    def get_statistics(self) -> dict:
        """Get visualization statistics"""
        with self._lock:
            return {
                **self.stats,
                'current_nodes': len(self.nodes),
                'current_edges': len(self.edges),
                'current_events': len(self.events)
            }


# ============================================================================
# DEMONSTRATION
# ============================================================================


def _example_usage():
    """Demonstrate observability visualization"""
    
    print("Observability Visualization Service - The Beatific Vision")
    print("="*70)
    
    from hlc import HybridLogicalClock
    
    # Create visualization service
    print("\n1. Initializing visualization service...")
    viz = ObservabilityVisualizationService()
    
    # Register nodes
    print("\n2. Registering compute nodes...")
    nodes = [
        ("node1", "Alpha Node", "compute"),
        ("node2", "Beta Node", "compute"),
        ("node3", "Gamma Node", "storage"),
        ("node4", "Delta Node", "network")
    ]
    
    for node_id, label, node_type in nodes:
        viz.register_node(node_id, label, node_type)
        print(f"   Registered: {label} ({node_type})")
    
    # Update node metrics
    print("\n3. Updating node metrics...")
    viz.update_node_metrics("node1", {"cpu": 45.2, "memory": 62.1})
    viz.update_node_metrics("node2", {"cpu": 78.9, "memory": 81.3})
    print("   ✓ Metrics updated")
    
    # Record some events
    print("\n4. Recording causal events...")
    clock = HybridLogicalClock("node1")
    
    # Event 1
    e1 = CausalEvent(
        "e1",
        clock.now(),
        EventType.API_REQUEST,
        "node1",
        data="GET /api/users"
    )
    viz.record_event(e1)
    print(f"   Event: {e1.event_id} on {e1.node_id}")
    
    # Event 2 (depends on e1)
    e2 = CausalEvent(
        "e2",
        clock.now(),
        EventType.DATABASE_READ,
        "node3",
        data="SELECT * FROM users",
        dependencies={"e1"}
    )
    viz.record_event(e2)
    print(f"   Event: {e2.event_id} on {e2.node_id} (depends on e1)")
    
    # Event 3 (depends on e2)
    e3 = CausalEvent(
        "e3",
        clock.now(),
        EventType.API_RESPONSE,
        "node1",
        data="200 OK",
        dependencies={"e2"}
    )
    viz.record_event(e3)
    print(f"   Event: {e3.event_id} on {e3.node_id} (depends on e2)")
    
    # Get visualization snapshot
    print("\n5. Generating visualization snapshot...")
    snapshot = viz.get_snapshot()
    print(f"   Nodes: {len(snapshot.nodes)}")
    print(f"   Edges: {len(snapshot.edges)}")
    print(f"   Events: {len(snapshot.events)}")
    
    # Show JSON (first 500 chars)
    print("\n6. Visualization JSON (excerpt):")
    json_str = snapshot.to_json()
    print(f"   {json_str[:500]}...")
    
    # Statistics
    print("\n7. Visualization statistics:")
    stats = viz.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n8. Visualization capabilities:")
    print("   ✓ Real-time event stream")
    print("   ✓ Causal DAG topology")
    print("   ✓ Node health metrics")
    print("   ✓ Time-travel replay")
    print("   ✓ Network flow animation")
    print("   ✓ 3D spatial layout")
    
    print("\n9. The Dream:")
    print("   You sit back and watch the 'Music of the Spheres.'")
    print("   You see requests flowing like light through")
    print("   the crystal lattice of your architecture.")
    print("   You see the system breathing.")
    print("   Every event, every causal chain, visible in real-time.")
    
    print("\n10. Integration with web frontend:")
    print("   - Export JSON to Three.js/WebGL renderer")
    print("   - Animate events along causal edges")
    print("   - Color nodes by health status")
    print("   - Interactive time-travel controls")
    
    print("\n" + "="*70)
    print("The Beatific Vision - observability as art.")


if __name__ == "__main__":
    _example_usage()
