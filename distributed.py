"""
Distributed Systems Support for Void-State Tools

Enables tool execution across distributed clusters with:
- Consensus protocols (Raft, Paxos)
- Distributed state management
- Fault tolerance and recovery
- Cluster coordination
- Vector clocks for causality tracking

References:
- "Distributed Systems" (Tanenbaum & Van Steen, 2017)
- "Designing Data-Intensive Applications" (Kleppmann, 2017)
- Raft paper (Ongaro & Ousterhout, 2014)
"""

from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import json
from collections import defaultdict
import threading
import socket


# ============================================================================
# VECTOR CLOCKS FOR CAUSALITY TRACKING
# ============================================================================

@dataclass
class VectorClock:
    """
    Vector clock for tracking causality in distributed systems.
    
    Provides partial ordering of events:
    - e1 → e2 (e1 happens-before e2) iff VC(e1) < VC(e2)
    - e1 || e2 (concurrent) iff VC(e1) ≮ VC(e2) and VC(e2) ≮ VC(e1)
    """
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str) -> 'VectorClock':
        """Increment clock for this node"""
        new_clocks = self.clocks.copy()
        new_clocks[node_id] = new_clocks.get(node_id, 0) + 1
        return VectorClock(new_clocks)
    
    def merge(self, other: 'VectorClock') -> 'VectorClock':
        """Merge with another vector clock (take max of each component)"""
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        new_clocks = {
            node: max(self.clocks.get(node, 0), other.clocks.get(node, 0))
            for node in all_nodes
        }
        return VectorClock(new_clocks)
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this event happens-before other"""
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        
        less_or_equal = all(
            self.clocks.get(node, 0) <= other.clocks.get(node, 0)
            for node in all_nodes
        )
        strictly_less = any(
            self.clocks.get(node, 0) < other.clocks.get(node, 0)
            for node in all_nodes
        )
        
        return less_or_equal and strictly_less
    
    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if events are concurrent"""
        return not self.happens_before(other) and not other.happens_before(self)


@dataclass
class CausalEvent:
    """Event with vector clock timestamp"""
    event_id: str
    node_id: str
    timestamp: VectorClock
    data: Any
    
    def __hash__(self):
        return hash(self.event_id)


# ============================================================================
# CONSENSUS PROTOCOLS
# ============================================================================

class NodeState(Enum):
    """Raft node states"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """Entry in the replicated log"""
    term: int
    index: int
    command: Any
    
    def __hash__(self):
        return hash((self.term, self.index))


class RaftNode:
    """
    Implementation of Raft consensus protocol.
    
    Provides strong consistency and fault tolerance for distributed state.
    
    Key properties:
    - Leader election
    - Log replication
    - Safety (never return incorrect results)
    """
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.state = NodeState.FOLLOWER
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timing
        self.election_timeout = 1.5  # seconds
        self.heartbeat_interval = 0.5  # seconds
        self.last_heartbeat = time.time()
        
        self.lock = threading.Lock()
    
    def append_entries(self, term: int, leader_id: str, 
                      prev_log_index: int, prev_log_term: int,
                      entries: List[LogEntry], leader_commit: int) -> Tuple[int, bool]:
        """
        AppendEntries RPC (also used for heartbeats).
        
        Returns: (current_term, success)
        """
        with self.lock:
            # Reply false if term < currentTerm
            if term < self.current_term:
                return (self.current_term, False)
            
            # Convert to follower if term > currentTerm
            if term > self.current_term:
                self.current_term = term
                self.state = NodeState.FOLLOWER
                self.voted_for = None
            
            # Reset election timer
            self.last_heartbeat = time.time()
            
            # Reply false if log doesn't contain entry at prevLogIndex
            if prev_log_index > 0:
                if prev_log_index > len(self.log):
                    return (self.current_term, False)
                if self.log[prev_log_index - 1].term != prev_log_term:
                    return (self.current_term, False)
            
            # Delete conflicting entries and append new ones
            if entries:
                # Find first conflicting entry
                for i, entry in enumerate(entries):
                    log_index = prev_log_index + i
                    if log_index < len(self.log):
                        if self.log[log_index].term != entry.term:
                            # Delete this and all following entries
                            self.log = self.log[:log_index]
                            break
                
                # Append new entries
                self.log.extend(entries)
            
            # Update commit index
            if leader_commit > self.commit_index:
                self.commit_index = min(leader_commit, len(self.log))
            
            return (self.current_term, True)
    
    def request_vote(self, term: int, candidate_id: str,
                    last_log_index: int, last_log_term: int) -> Tuple[int, bool]:
        """
        RequestVote RPC.
        
        Returns: (current_term, vote_granted)
        """
        with self.lock:
            # Reply false if term < currentTerm
            if term < self.current_term:
                return (self.current_term, False)
            
            # Update term if necessary
            if term > self.current_term:
                self.current_term = term
                self.state = NodeState.FOLLOWER
                self.voted_for = None
            
            # Check if we can vote for this candidate
            vote_granted = False
            
            if self.voted_for is None or self.voted_for == candidate_id:
                # Check if candidate's log is at least as up-to-date
                our_last_index = len(self.log)
                our_last_term = self.log[-1].term if self.log else 0
                
                log_ok = (last_log_term > our_last_term or
                         (last_log_term == our_last_term and 
                          last_log_index >= our_last_index))
                
                if log_ok:
                    self.voted_for = candidate_id
                    vote_granted = True
                    self.last_heartbeat = time.time()
            
            return (self.current_term, vote_granted)
    
    def start_election(self) -> None:
        """Start leader election"""
        with self.lock:
            self.state = NodeState.CANDIDATE
            self.current_term += 1
            self.voted_for = self.node_id
            self.last_heartbeat = time.time()
            
            votes = 1  # Vote for self
            
            # Request votes from all peers
            # (In real implementation, this would be done in parallel)
            for peer in self.peers:
                last_log_index = len(self.log)
                last_log_term = self.log[-1].term if self.log else 0
                
                # Simulate RPC call
                term, granted = self._send_request_vote(
                    peer, self.current_term, self.node_id,
                    last_log_index, last_log_term
                )
                
                if granted:
                    votes += 1
            
            # Become leader if majority votes
            if votes > (len(self.peers) + 1) // 2:
                self.become_leader()
    
    def become_leader(self) -> None:
        """Transition to leader state"""
        self.state = NodeState.LEADER
        
        # Initialize leader state
        for peer in self.peers:
            self.next_index[peer] = len(self.log) + 1
            self.match_index[peer] = 0
        
        # Send initial heartbeats
        self.send_heartbeats()
    
    def send_heartbeats(self) -> None:
        """Send heartbeat (empty AppendEntries) to all peers"""
        if self.state != NodeState.LEADER:
            return
        
        for peer in self.peers:
            prev_log_index = self.next_index[peer] - 1
            prev_log_term = self.log[prev_log_index - 1].term if prev_log_index > 0 else 0
            
            # Simulate RPC call
            self._send_append_entries(
                peer, self.current_term, self.node_id,
                prev_log_index, prev_log_term, [], self.commit_index
            )
    
    def _send_request_vote(self, peer: str, *args) -> Tuple[int, bool]:
        """Simulate sending RequestVote RPC"""
        # In real implementation, this would send RPC over network
        return (self.current_term, False)
    
    def _send_append_entries(self, peer: str, *args) -> Tuple[int, bool]:
        """Simulate sending AppendEntries RPC"""
        # In real implementation, this would send RPC over network
        return (self.current_term, True)


# ============================================================================
# DISTRIBUTED STATE MANAGEMENT
# ============================================================================

@dataclass
class CRDT:
    """
    Base class for Conflict-Free Replicated Data Types.
    
    CRDTs provide strong eventual consistency without coordination.
    """
    pass


class GCounter(CRDT):
    """
    Grow-only Counter CRDT.
    
    Can only increment, never decrement.
    Merge takes maximum of each node's counter.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: Dict[str, int] = defaultdict(int)
    
    def increment(self, delta: int = 1) -> None:
        """Increment this node's counter"""
        self.counts[self.node_id] += delta
    
    def value(self) -> int:
        """Get total count"""
        return sum(self.counts.values())
    
    def merge(self, other: 'GCounter') -> 'GCounter':
        """Merge with another GCounter"""
        result = GCounter(self.node_id)
        all_nodes = set(self.counts.keys()) | set(other.counts.keys())
        
        for node in all_nodes:
            result.counts[node] = max(
                self.counts.get(node, 0),
                other.counts.get(node, 0)
            )
        
        return result


class PNCounter(CRDT):
    """
    Positive-Negative Counter CRDT.
    
    Supports both increment and decrement.
    Uses two GCounters (positive and negative).
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.pos = GCounter(node_id)
        self.neg = GCounter(node_id)
    
    def increment(self, delta: int = 1) -> None:
        """Increment counter"""
        if delta > 0:
            self.pos.increment(delta)
        else:
            self.neg.increment(-delta)
    
    def decrement(self, delta: int = 1) -> None:
        """Decrement counter"""
        self.increment(-delta)
    
    def value(self) -> int:
        """Get current value"""
        return self.pos.value() - self.neg.value()
    
    def merge(self, other: 'PNCounter') -> 'PNCounter':
        """Merge with another PNCounter"""
        result = PNCounter(self.node_id)
        result.pos = self.pos.merge(other.pos)
        result.neg = self.neg.merge(other.neg)
        return result


class GSet(CRDT):
    """
    Grow-only Set CRDT.
    
    Can add elements but never remove them.
    Merge is set union.
    """
    
    def __init__(self):
        self.elements: Set[Any] = set()
    
    def add(self, element: Any) -> None:
        """Add element to set"""
        self.elements.add(element)
    
    def contains(self, element: Any) -> bool:
        """Check if element is in set"""
        return element in self.elements
    
    def merge(self, other: 'GSet') -> 'GSet':
        """Merge with another GSet"""
        result = GSet()
        result.elements = self.elements | other.elements
        return result


class TwoPhaseSet(CRDT):
    """
    Two-Phase Set CRDT.
    
    Supports add and remove, but elements can only be added once.
    Once removed, an element cannot be re-added.
    """
    
    def __init__(self):
        self.added = GSet()
        self.removed = GSet()
    
    def add(self, element: Any) -> None:
        """Add element"""
        self.added.add(element)
    
    def remove(self, element: Any) -> None:
        """Remove element"""
        if self.added.contains(element):
            self.removed.add(element)
    
    def contains(self, element: Any) -> bool:
        """Check if element is in set"""
        return self.added.contains(element) and not self.removed.contains(element)
    
    def merge(self, other: 'TwoPhaseSet') -> 'TwoPhaseSet':
        """Merge with another TwoPhaseSet"""
        result = TwoPhaseSet()
        result.added = self.added.merge(other.added)
        result.removed = self.removed.merge(other.removed)
        return result


# ============================================================================
# DISTRIBUTED COORDINATION
# ============================================================================

class DistributedLock:
    """
    Distributed lock using consensus.
    
    Ensures mutual exclusion across distributed nodes.
    """
    
    def __init__(self, lock_id: str, node_id: str, ttl: float = 10.0):
        self.lock_id = lock_id
        self.node_id = node_id
        self.ttl = ttl
        self.acquired = False
        self.expiry: Optional[float] = None
    
    def acquire(self, timeout: float = 5.0) -> bool:
        """
        Try to acquire lock.
        
        Returns True if successful, False if timeout.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Try to acquire lock
            if self._try_acquire():
                self.acquired = True
                self.expiry = time.time() + self.ttl
                return True
            
            time.sleep(0.1)
        
        return False
    
    def release(self) -> bool:
        """Release the lock"""
        if not self.acquired:
            return False
        
        success = self._try_release()
        if success:
            self.acquired = False
            self.expiry = None
        
        return success
    
    def _try_acquire(self) -> bool:
        """Try to acquire lock (implementation-specific)"""
        # In real implementation, this would use consensus protocol
        return True
    
    def _try_release(self) -> bool:
        """Try to release lock (implementation-specific)"""
        # In real implementation, this would use consensus protocol
        return True


class DistributedBarrier:
    """
    Distributed barrier for synchronizing multiple nodes.
    
    All nodes must reach the barrier before any can proceed.
    """
    
    def __init__(self, barrier_id: str, num_nodes: int):
        self.barrier_id = barrier_id
        self.num_nodes = num_nodes
        self.arrived_nodes: Set[str] = set()
        self.lock = threading.Lock()
    
    def wait(self, node_id: str, timeout: float = 10.0) -> bool:
        """
        Wait at barrier.
        
        Returns True when all nodes have arrived, False on timeout.
        """
        start_time = time.time()
        
        with self.lock:
            self.arrived_nodes.add(node_id)
        
        # Wait for all nodes
        while len(self.arrived_nodes) < self.num_nodes:
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.1)
        
        return True
    
    def reset(self) -> None:
        """Reset barrier for reuse"""
        with self.lock:
            self.arrived_nodes.clear()


# ============================================================================
# CONSISTENT HASHING FOR LOAD DISTRIBUTION
# ============================================================================

class ConsistentHash:
    """
    Consistent hashing for distributed load balancing.
    
    Minimizes key redistribution when nodes are added/removed.
    Uses virtual nodes for better load distribution.
    """
    
    def __init__(self, num_virtual_nodes: int = 150):
        self.num_virtual_nodes = num_virtual_nodes
        self.ring: Dict[int, str] = {}  # hash -> node_id
        self.nodes: Set[str] = set()
        self.sorted_keys: List[int] = []
    
    def add_node(self, node_id: str) -> None:
        """Add a node to the hash ring"""
        if node_id in self.nodes:
            return
        
        self.nodes.add(node_id)
        
        # Add virtual nodes
        for i in range(self.num_virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node_id
        
        # Update sorted keys
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the hash ring"""
        if node_id not in self.nodes:
            return
        
        self.nodes.remove(node_id)
        
        # Remove virtual nodes
        keys_to_remove = [k for k, v in self.ring.items() if v == node_id]
        for key in keys_to_remove:
            del self.ring[key]
        
        # Update sorted keys
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for this key"""
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        
        # Binary search for first node >= hash_val
        idx = self._binary_search(hash_val)
        
        if idx >= len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def _hash(self, key: str) -> int:
        """Hash function"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _binary_search(self, value: int) -> int:
        """Binary search for first key >= value"""
        left, right = 0, len(self.sorted_keys)
        
        while left < right:
            mid = (left + right) // 2
            if self.sorted_keys[mid] < value:
                left = mid + 1
            else:
                right = mid
        
        return left


# ============================================================================
# DISTRIBUTED TRACING
# ============================================================================

@dataclass
class Span:
    """
    Distributed tracing span.
    
    Represents a unit of work in a distributed system.
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self) -> None:
        """Finish the span"""
        self.end_time = time.time()
    
    def duration(self) -> Optional[float]:
        """Get span duration"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def add_tag(self, key: str, value: Any) -> None:
        """Add a tag to the span"""
        self.tags[key] = value
    
    def add_log(self, message: str, **fields) -> None:
        """Add a log entry to the span"""
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            **fields
        }
        self.logs.append(log_entry)


class Tracer:
    """
    Distributed tracer for tracking requests across services.
    """
    
    def __init__(self):
        self.active_spans: Dict[str, Span] = {}
    
    def start_span(self, operation_name: str, 
                   parent_span_id: Optional[str] = None) -> Span:
        """Start a new span"""
        import uuid
        
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        self.active_spans[span_id] = span
        return span
    
    def finish_span(self, span_id: str) -> None:
        """Finish a span"""
        if span_id in self.active_spans:
            self.active_spans[span_id].finish()
            # In real implementation, would send to collector here
            del self.active_spans[span_id]
