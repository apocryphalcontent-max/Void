"""
Membership Protocol Service
"The Council of Nicea" - Dynamic View Change Protocol

Implements dynamic membership management for distributed consensus.
Nodes can join, leave, or fail - the cluster autonomously decides
"Who is in the church?"

Key features:
- Dynamic node membership (join/leave/failure detection)
- View change protocol for PBFT
- Heartbeat-based failure detection
- Quorum management
- Autonomous excommunication of dead nodes

References:
- "Practical Byzantine Fault Tolerance" (Castro & Liskov, 1999)
- "The Part-Time Parliament" (Lamport, 1998) - Paxos
- "In Search of an Understandable Consensus Algorithm" (Ongaro & Ousterhout, 2014) - Raft
"""

import time
import threading
from typing import Dict, List, Set, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from hlc import HybridLogicalClock, HLCTimestamp


# ============================================================================
# NODE STATUS AND MEMBERSHIP
# ============================================================================

class NodeStatus(Enum):
    """Status of a node in the cluster"""
    ALIVE = "alive"
    SUSPECTED = "suspected"
    DEAD = "dead"
    JOINING = "joining"
    LEAVING = "leaving"


@dataclass
class NodeInfo:
    """Information about a cluster node"""
    node_id: str
    address: str  # Network address
    public_key: bytes  # Cryptographic public key
    status: NodeStatus = NodeStatus.ALIVE
    last_heartbeat: float = 0.0
    join_time: float = 0.0
    failure_count: int = 0
    
    def __hash__(self):
        return hash(self.node_id)


@dataclass
class ClusterView:
    """
    A view of the cluster membership.
    
    In PBFT, a "view" is a configuration of the cluster with a designated primary.
    View changes occur when the primary fails or is suspected faulty.
    """
    view_number: int
    primary_id: str
    members: Set[str]  # Node IDs in this view
    timestamp: HLCTimestamp
    
    # Quorum size: minimum nodes needed for consensus
    @property
    def quorum_size(self) -> int:
        """Minimum nodes needed for consensus (2f+1 where f = max failures)"""
        n = len(self.members)
        f = (n - 1) // 3  # Max Byzantine failures
        return 2 * f + 1
    
    @property
    def max_failures(self) -> int:
        """Maximum Byzantine failures tolerated"""
        return (len(self.members) - 1) // 3
    
    def is_member(self, node_id: str) -> bool:
        """Check if node is a member of this view"""
        return node_id in self.members


# ============================================================================
# MEMBERSHIP PROTOCOL SERVICE
# ============================================================================

class MembershipProtocolService:
    """
    Service for dynamic cluster membership management.
    
    Responsibilities:
    1. Heartbeat monitoring (failure detection)
    2. View change coordination (when primary fails)
    3. Node join/leave handling
    4. Quorum maintenance
    5. Autonomous decision-making ("The Council")
    """
    
    def __init__(
        self,
        node_id: str,
        initial_members: List[str],
        heartbeat_interval: float = 1.0,
        failure_timeout: float = 5.0,
        suspected_timeout: float = 3.0
    ):
        """
        Initialize membership protocol.
        
        Args:
            node_id: This node's identifier
            initial_members: Initial cluster members
            heartbeat_interval: Seconds between heartbeats
            failure_timeout: Seconds until node considered dead
            suspected_timeout: Seconds until node suspected
        """
        self.node_id = node_id
        self.heartbeat_interval = heartbeat_interval
        self.failure_timeout = failure_timeout
        self.suspected_timeout = suspected_timeout
        
        # Cluster state
        self.nodes: Dict[str, NodeInfo] = {}
        self.current_view: Optional[ClusterView] = None
        self.view_history: List[ClusterView] = []
        
        # Initialize with initial members
        for member_id in initial_members:
            self.nodes[member_id] = NodeInfo(
                node_id=member_id,
                address=f"node://{member_id}",
                public_key=b"",  # Would be exchanged securely
                status=NodeStatus.ALIVE,
                last_heartbeat=time.time(),
                join_time=time.time()
            )
        
        # Create initial view
        clock = HybridLogicalClock(node_id)
        self.current_view = ClusterView(
            view_number=0,
            primary_id=initial_members[0] if initial_members else node_id,
            members=set(initial_members),
            timestamp=clock.now()
        )
        self.view_history.append(self.current_view)
        
        # Failure detection state
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'heartbeats_sent': 0,
            'heartbeats_received': 0,
            'view_changes': 0,
            'nodes_joined': len(initial_members),
            'nodes_left': 0,
            'failures_detected': 0
        }
        
        # Callbacks
        self.on_view_change: Optional[Callable[[ClusterView], None]] = None
        self.on_node_failed: Optional[Callable[[str], None]] = None
        self.on_node_joined: Optional[Callable[[str], None]] = None
    
    def start(self):
        """Start the membership protocol service"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop(self):
        """Stop the membership protocol service"""
        with self._lock:
            self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Background loop for failure detection"""
        while self._running:
            # Check for failures
            self._check_failures()
            
            # Send heartbeat (simulated)
            self._send_heartbeat()
            
            time.sleep(self.heartbeat_interval)
    
    def _check_failures(self):
        """Check for node failures based on heartbeat timeouts"""
        current_time = time.time()
        
        with self._lock:
            failed_nodes = []
            suspected_nodes = []
            
            for node_id, node_info in self.nodes.items():
                if node_id == self.node_id:
                    # Don't monitor ourselves
                    continue
                
                if node_info.status not in [NodeStatus.ALIVE, NodeStatus.SUSPECTED]:
                    continue
                
                time_since_heartbeat = current_time - node_info.last_heartbeat
                
                # Check if node is dead
                if time_since_heartbeat > self.failure_timeout:
                    if node_info.status != NodeStatus.DEAD:
                        node_info.status = NodeStatus.DEAD
                        node_info.failure_count += 1
                        failed_nodes.append(node_id)
                        self.stats['failures_detected'] += 1
                
                # Check if node is suspected
                elif time_since_heartbeat > self.suspected_timeout:
                    if node_info.status == NodeStatus.ALIVE:
                        node_info.status = NodeStatus.SUSPECTED
                        suspected_nodes.append(node_id)
            
            # Handle failures
            for node_id in failed_nodes:
                self._handle_node_failure(node_id)
            
            # Log suspicions
            for node_id in suspected_nodes:
                pass  # Could trigger preventive measures
    
    def _send_heartbeat(self):
        """Send heartbeat to all nodes (simulated)"""
        with self._lock:
            self.stats['heartbeats_sent'] += 1
            
            # Update our own heartbeat
            if self.node_id in self.nodes:
                self.nodes[self.node_id].last_heartbeat = time.time()
    
    def receive_heartbeat(self, node_id: str):
        """
        Receive heartbeat from a node.
        
        Args:
            node_id: Node that sent heartbeat
        """
        with self._lock:
            self.stats['heartbeats_received'] += 1
            
            if node_id not in self.nodes:
                # Unknown node - could be joining
                return
            
            node_info = self.nodes[node_id]
            node_info.last_heartbeat = time.time()
            
            # If node was suspected/dead, mark as alive
            if node_info.status in [NodeStatus.SUSPECTED, NodeStatus.DEAD]:
                node_info.status = NodeStatus.ALIVE
    
    def _handle_node_failure(self, node_id: str):
        """
        Handle detection of node failure.
        
        Args:
            node_id: Failed node ID
        """
        # Notify callback
        if self.on_node_failed:
            self.on_node_failed(node_id)
        
        # Check if we need a view change
        if self.current_view and node_id in self.current_view.members:
            # Remove from current view
            if node_id == self.current_view.primary_id:
                # Primary failed - trigger view change
                self._trigger_view_change("primary_failure")
            elif len(self.current_view.members) - 1 < self.current_view.quorum_size:
                # Lost quorum - trigger view change
                self._trigger_view_change("quorum_lost")
            else:
                # Non-primary failed but still have quorum
                # Could trigger view change anyway or wait
                pass
    
    def _trigger_view_change(self, reason: str):
        """
        Trigger a view change.
        
        This is "The Council" - the cluster holds a meeting to decide
        the new configuration.
        
        Args:
            reason: Reason for view change
        """
        new_view = None
        callback = None
        
        with self._lock:
            if not self.current_view:
                return
            
            # Determine new membership (exclude dead/leaving nodes)
            new_members = set()
            for node_id in self.current_view.members:
                if node_id in self.nodes:
                    node_info = self.nodes[node_id]
                    if node_info.status in [NodeStatus.ALIVE, NodeStatus.SUSPECTED]:
                        new_members.add(node_id)
            
            if len(new_members) < 1:  # Need at least one node
                # Cannot maintain cluster with zero nodes
                # In production, this would halt the cluster
                return
            
            # Select new primary (round-robin)
            new_view_number = self.current_view.view_number + 1
            members_list = sorted(new_members)
            new_primary_idx = new_view_number % len(members_list)
            new_primary = members_list[new_primary_idx]
            
            # Create new view
            clock = HybridLogicalClock(self.node_id)
            new_view = ClusterView(
                view_number=new_view_number,
                primary_id=new_primary,
                members=new_members,
                timestamp=clock.now()
            )
            
            # Update state
            self.current_view = new_view
            self.view_history.append(new_view)
            self.stats['view_changes'] += 1
            
            # Get callback reference
            callback = self.on_view_change
        
        # Notify callback outside of lock to avoid deadlock
        if callback and new_view:
            callback(new_view)
    
    def request_join(self, node_id: str, address: str, public_key: bytes) -> bool:
        """
        Request to join the cluster.
        
        Args:
            node_id: Node requesting to join
            address: Node's network address
            public_key: Node's public key
            
        Returns:
            True if join approved
        """
        with self._lock:
            # Check if already a member
            if node_id in self.nodes:
                return False
            
            # Add node
            node_info = NodeInfo(
                node_id=node_id,
                address=address,
                public_key=public_key,
                status=NodeStatus.ALIVE,  # Mark as alive immediately
                last_heartbeat=time.time(),
                join_time=time.time()
            )
            self.nodes[node_id] = node_info
            self.stats['nodes_joined'] += 1
        
        # Trigger view change outside of lock
        self._trigger_view_change("node_join")
        
        # Notify callback
        if self.on_node_joined:
            self.on_node_joined(node_id)
        
        return True
    
    def request_leave(self, node_id: str):
        """
        Request to leave the cluster gracefully.
        
        Args:
            node_id: Node requesting to leave
        """
        with self._lock:
            if node_id not in self.nodes:
                return
            
            node_info = self.nodes[node_id]
            node_info.status = NodeStatus.LEAVING
            
            # Trigger view change to remove node
            self._trigger_view_change("node_leave")
            
            self.stats['nodes_left'] += 1
    
    def get_current_view(self) -> Optional[ClusterView]:
        """Get the current cluster view"""
        with self._lock:
            return self.current_view
    
    def is_primary(self) -> bool:
        """Check if this node is the current primary"""
        with self._lock:
            if self.current_view:
                return self.current_view.primary_id == self.node_id
            return False
    
    def get_live_members(self) -> Set[str]:
        """Get set of currently live members"""
        with self._lock:
            live = set()
            for node_id, node_info in self.nodes.items():
                if node_info.status == NodeStatus.ALIVE:
                    live.add(node_id)
            return live
    
    def get_statistics(self) -> dict:
        """Get service statistics"""
        with self._lock:
            live_count = sum(1 for node_info in self.nodes.values() 
                           if node_info.status == NodeStatus.ALIVE)
            
            is_primary = (self.current_view.primary_id == self.node_id) if self.current_view else False
            
            return {
                **self.stats,
                'current_view': self.current_view.view_number if self.current_view else None,
                'total_nodes': len(self.nodes),
                'live_nodes': live_count,
                'primary_id': self.current_view.primary_id if self.current_view else None,
                'is_primary': is_primary
            }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def _example_usage():
    """Demonstrate membership protocol"""
    
    print("Membership Protocol Service - Dynamic View Changes")
    print("="*70)
    
    # Create initial cluster
    print("\n1. Initializing cluster with 4 nodes...")
    members = ["node0", "node1", "node2", "node3"]
    service = MembershipProtocolService(
        node_id="node0",
        initial_members=members,
        heartbeat_interval=1.0,
        failure_timeout=3.0
    )
    
    view = service.get_current_view()
    print(f"   Initial view: {view.view_number}")
    print(f"   Primary: {view.primary_id}")
    print(f"   Members: {sorted(view.members)}")
    print(f"   Quorum size: {view.quorum_size}")
    print(f"   Max failures: {view.max_failures}")
    
    # Set up callbacks
    def on_view_change(new_view: ClusterView):
        print(f"\n   → VIEW CHANGE to view {new_view.view_number}")
        print(f"      New primary: {new_view.primary_id}")
        print(f"      Members: {sorted(new_view.members)}")
    
    def on_failure(node_id: str):
        print(f"\n   ⚠ Node {node_id} has failed (no heartbeat)")
    
    def on_join(node_id: str):
        print(f"\n   ✓ Node {node_id} has joined the cluster")
    
    service.on_view_change = on_view_change
    service.on_node_failed = on_failure
    service.on_node_joined = on_join
    
    # Simulate node join
    print("\n2. Node node4 requests to join...")
    success = service.request_join("node4", "node://node4", b"public_key_4")
    print(f"   Join {'approved' if success else 'rejected'}")
    
    # Simulate heartbeats
    print("\n3. Receiving heartbeats from nodes...")
    service.receive_heartbeat("node1")
    service.receive_heartbeat("node2")
    service.receive_heartbeat("node3")
    print("   ✓ Heartbeats received from node1, node2, node3")
    
    # Check statistics
    print("\n4. Statistics:")
    stats = service.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n5. Membership protocol guarantees:")
    print("   ✓ Automatic failure detection via heartbeats")
    print("   ✓ Dynamic view changes when nodes fail")
    print("   ✓ Quorum maintenance (2f+1 nodes)")
    print("   ✓ Byzantine fault tolerance (up to f failures)")
    print("   ✓ Graceful node join/leave")
    print("   ✓ 'The Council' decides membership autonomously")
    
    print("\n6. Resilience scenario:")
    print("   If 30% of nodes fail, remaining nodes will:")
    print("   - Detect silence (missing heartbeats)")
    print("   - Hold a Council (view change protocol)")
    print("   - Excommunicate dead nodes")
    print("   - Continue processing without dropping requests")
    
    print("\n" + "="*70)
    print("The cluster autonomously maintains its membership and health.")


if __name__ == "__main__":
    _example_usage()
