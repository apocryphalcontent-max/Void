"""
PBFT (Practical Byzantine Fault Tolerance) Consensus

High-level Python coordination layer for PBFT consensus protocol.
The core cryptographic operations and message handling will eventually
be moved to Rust for performance.

PBFT provides Byzantine fault tolerance with:
- Safety: Never returns incorrect results
- Liveness: Eventually makes progress
- f < n/3: Tolerates up to f faulty nodes out of n total

Protocol phases:
1. Pre-prepare: Primary proposes value
2. Prepare: Nodes validate and vote
3. Commit: Nodes commit after 2f+1 prepare votes
4. Reply: Return result to client

References:
- "Practical Byzantine Fault Tolerance" (Castro & Liskov, 1999)
- "Byzantine Fault Tolerance" (Cachin et al., 2011)
"""

from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import json
from collections import defaultdict
import threading

from hlc import HybridLogicalClock, HLCTimestamp


# ============================================================================
# MESSAGE TYPES
# ============================================================================

class MessageType(Enum):
    """PBFT message types"""
    REQUEST = "request"
    PRE_PREPARE = "pre-prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    REPLY = "reply"
    VIEW_CHANGE = "view-change"
    NEW_VIEW = "new-view"
    CHECKPOINT = "checkpoint"


@dataclass
class PBFTMessage:
    """
    A PBFT protocol message.
    
    All messages include:
    - msg_type: Type of message
    - view: Current view number
    - sequence: Sequence number
    - node_id: Sender node
    - timestamp: HLC timestamp
    - digest: Hash of the operation
    - signature: Cryptographic signature (placeholder)
    """
    msg_type: MessageType
    view: int
    sequence: int
    node_id: str
    timestamp: HLCTimestamp
    digest: str
    payload: Any = None
    signature: Optional[str] = None
    
    def __hash__(self):
        return hash((self.msg_type, self.view, self.sequence, self.node_id))
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'msg_type': self.msg_type.value,
            'view': self.view,
            'sequence': self.sequence,
            'node_id': self.node_id,
            'timestamp': self.timestamp.to_dict(),
            'digest': self.digest,
            'payload': self.payload,
            'signature': self.signature
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PBFTMessage':
        """Deserialize from dictionary"""
        return cls(
            msg_type=MessageType(data['msg_type']),
            view=data['view'],
            sequence=data['sequence'],
            node_id=data['node_id'],
            timestamp=HLCTimestamp.from_dict(data['timestamp']),
            digest=data['digest'],
            payload=data.get('payload'),
            signature=data.get('signature')
        )


# ============================================================================
# PBFT NODE STATE
# ============================================================================

class NodeState(Enum):
    """PBFT node states"""
    NORMAL = "normal"
    VIEW_CHANGE = "view-change"
    RECOVERY = "recovery"


@dataclass
class ConsensusState:
    """State for a consensus instance (sequence number)"""
    sequence: int
    pre_prepare: Optional[PBFTMessage] = None
    prepares: Set[str] = field(default_factory=set)  # Node IDs that sent prepare
    commits: Set[str] = field(default_factory=set)   # Node IDs that sent commit
    prepared: bool = False
    committed: bool = False
    result: Optional[Any] = None


class PBFTNode:
    """
    PBFT consensus node.
    
    This is a high-level Python implementation. Performance-critical
    operations (especially cryptographic signatures and message serialization)
    should be moved to Rust.
    """
    
    def __init__(
        self,
        node_id: str,
        replica_ids: List[str],
        f: int,
        checkpoint_interval: int = 100
    ):
        """
        Initialize PBFT node.
        
        Args:
            node_id: This node's identifier
            replica_ids: All replica identifiers (including self)
            f: Maximum number of faulty nodes (n = 3f + 1)
            checkpoint_interval: Sequence numbers between checkpoints
        """
        self.node_id = node_id
        self.replica_ids = replica_ids
        self.f = f
        self.n = len(replica_ids)
        self.checkpoint_interval = checkpoint_interval
        
        # Validate configuration
        if self.n < 3 * f + 1:
            raise ValueError(f"Need at least {3*f + 1} replicas for f={f}")
        
        # Protocol state
        self.view = 0
        self.sequence = 0
        self.state = NodeState.NORMAL
        
        # Consensus state per sequence
        self.consensus: Dict[int, ConsensusState] = {}
        
        # Message log
        self.message_log: List[PBFTMessage] = []
        
        # Checkpoints
        self.checkpoints: Dict[int, str] = {}  # sequence -> state digest
        self.stable_checkpoint = 0
        
        # HLC for timestamps
        self.clock = HybridLogicalClock(node_id)
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'view_changes': 0,
            'messages_sent': 0,
            'messages_received': 0
        }
        
        self._lock = threading.Lock()
        
        # Callback for sending messages (set by user)
        self.send_message: Optional[Callable[[str, PBFTMessage], None]] = None
        
        # Callback for applying operations (set by user)
        self.apply_operation: Optional[Callable[[Any], Any]] = None
    
    def is_primary(self, view: Optional[int] = None) -> bool:
        """Check if this node is the primary for the view"""
        if view is None:
            view = self.view
        primary_idx = view % self.n
        return self.replica_ids[primary_idx] == self.node_id
    
    def get_primary(self, view: Optional[int] = None) -> str:
        """Get the primary node ID for the view"""
        if view is None:
            view = self.view
        primary_idx = view % self.n
        return self.replica_ids[primary_idx]
    
    def request(self, operation: Any) -> Optional[Any]:
        """
        Submit a client request (if this is the primary).
        
        Args:
            operation: Operation to execute
            
        Returns:
            Result if committed, None otherwise
        """
        with self._lock:
            if not self.is_primary():
                # Forward to primary
                return None
            
            # Assign sequence number
            self.sequence += 1
            seq = self.sequence
            
            # Compute digest
            digest = self._compute_digest(operation)
            
            # Create pre-prepare message
            timestamp = self.clock.now()
            pre_prepare = PBFTMessage(
                msg_type=MessageType.PRE_PREPARE,
                view=self.view,
                sequence=seq,
                node_id=self.node_id,
                timestamp=timestamp,
                digest=digest,
                payload=operation
            )
            
            # Sign message (placeholder)
            pre_prepare.signature = self._sign_message(pre_prepare)
            
            # Store in log
            self.message_log.append(pre_prepare)
            
            # Initialize consensus state
            self.consensus[seq] = ConsensusState(sequence=seq, pre_prepare=pre_prepare)
            
            # Broadcast to replicas
            self._broadcast_message(pre_prepare)
            
            # Primary also sends prepare
            self._send_prepare(seq, digest)
            
            return None  # Result comes later via commit
    
    def receive_pre_prepare(self, msg: PBFTMessage):
        """
        Receive a pre-prepare message.
        
        Args:
            msg: Pre-prepare message
        """
        with self._lock:
            self.stats['messages_received'] += 1
            
            # Validate message
            if not self._validate_pre_prepare(msg):
                return
            
            # Store in consensus state
            if msg.sequence not in self.consensus:
                self.consensus[msg.sequence] = ConsensusState(sequence=msg.sequence)
            
            state = self.consensus[msg.sequence]
            state.pre_prepare = msg
            
            # Send prepare
            self._send_prepare(msg.sequence, msg.digest)
    
    def receive_prepare(self, msg: PBFTMessage):
        """
        Receive a prepare message.
        
        Args:
            msg: Prepare message
        """
        with self._lock:
            self.stats['messages_received'] += 1
            
            # Validate message
            if not self._validate_prepare(msg):
                return
            
            # Store prepare
            if msg.sequence not in self.consensus:
                self.consensus[msg.sequence] = ConsensusState(sequence=msg.sequence)
            
            state = self.consensus[msg.sequence]
            state.prepares.add(msg.node_id)
            
            # Check if prepared (2f + 1 prepares)
            if len(state.prepares) >= 2 * self.f + 1 and not state.prepared:
                state.prepared = True
                
                # Send commit
                self._send_commit(msg.sequence, msg.digest)
    
    def receive_commit(self, msg: PBFTMessage):
        """
        Receive a commit message.
        
        Args:
            msg: Commit message
        """
        with self._lock:
            self.stats['messages_received'] += 1
            
            # Validate message
            if not self._validate_commit(msg):
                return
            
            # Store commit
            if msg.sequence not in self.consensus:
                return
            
            state = self.consensus[msg.sequence]
            state.commits.add(msg.node_id)
            
            # Check if committed (2f + 1 commits)
            if len(state.commits) >= 2 * self.f + 1 and not state.committed:
                state.committed = True
                
                # Execute operation
                if state.pre_prepare and self.apply_operation:
                    result = self.apply_operation(state.pre_prepare.payload)
                    state.result = result
                    
                    self.stats['requests_processed'] += 1
                    
                    # Check if we should checkpoint
                    if msg.sequence % self.checkpoint_interval == 0:
                        self._create_checkpoint(msg.sequence)
    
    def _send_prepare(self, sequence: int, digest: str):
        """Send prepare message"""
        timestamp = self.clock.now()
        prepare = PBFTMessage(
            msg_type=MessageType.PREPARE,
            view=self.view,
            sequence=sequence,
            node_id=self.node_id,
            timestamp=timestamp,
            digest=digest
        )
        prepare.signature = self._sign_message(prepare)
        
        self.message_log.append(prepare)
        self._broadcast_message(prepare)
    
    def _send_commit(self, sequence: int, digest: str):
        """Send commit message"""
        timestamp = self.clock.now()
        commit = PBFTMessage(
            msg_type=MessageType.COMMIT,
            view=self.view,
            sequence=sequence,
            node_id=self.node_id,
            timestamp=timestamp,
            digest=digest
        )
        commit.signature = self._sign_message(commit)
        
        self.message_log.append(commit)
        self._broadcast_message(commit)
    
    def _broadcast_message(self, msg: PBFTMessage):
        """Broadcast message to all replicas"""
        if not self.send_message:
            return
        
        for replica_id in self.replica_ids:
            if replica_id != self.node_id:
                self.send_message(replica_id, msg)
                self.stats['messages_sent'] += 1
    
    def _validate_pre_prepare(self, msg: PBFTMessage) -> bool:
        """Validate pre-prepare message"""
        # Check view matches
        if msg.view != self.view:
            return False
        
        # Check sender is primary
        if msg.node_id != self.get_primary(msg.view):
            return False
        
        # Check sequence is in valid range
        if msg.sequence <= self.stable_checkpoint:
            return False
        
        # Check signature (placeholder)
        # In production, verify Ed25519 signature
        
        return True
    
    def _validate_prepare(self, msg: PBFTMessage) -> bool:
        """Validate prepare message"""
        if msg.view != self.view:
            return False
        
        if msg.sequence <= self.stable_checkpoint:
            return False
        
        return True
    
    def _validate_commit(self, msg: PBFTMessage) -> bool:
        """Validate commit message"""
        if msg.view != self.view:
            return False
        
        if msg.sequence <= self.stable_checkpoint:
            return False
        
        return True
    
    def _compute_digest(self, operation: Any) -> str:
        """Compute message digest"""
        data = json.dumps(operation, sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()
    
    def _sign_message(self, msg: PBFTMessage) -> str:
        """
        Sign message (placeholder).
        
        In production, use Ed25519 signatures.
        This should be moved to Rust for performance.
        """
        data = json.dumps(msg.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()
    
    def _create_checkpoint(self, sequence: int):
        """Create checkpoint at sequence number"""
        # Compute state digest (placeholder)
        state_digest = hashlib.sha256(f"state_{sequence}".encode()).hexdigest()
        self.checkpoints[sequence] = state_digest
        self.stable_checkpoint = sequence
    
    def get_stats(self) -> dict:
        """Get node statistics"""
        with self._lock:
            return {
                **self.stats,
                'view': self.view,
                'sequence': self.sequence,
                'state': self.state.value,
                'consensus_instances': len(self.consensus),
                'stable_checkpoint': self.stable_checkpoint
            }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def _example_usage():
    """Demonstrate PBFT"""
    
    print("PBFT Consensus Protocol\n" + "="*70)
    
    # Create 4 replicas (f=1)
    replica_ids = ["node0", "node1", "node2", "node3"]
    f = 1
    
    nodes = {}
    for replica_id in replica_ids:
        node = PBFTNode(replica_id, replica_ids, f)
        nodes[replica_id] = node
    
    print(f"1. Created {len(replica_ids)} replicas (f={f})")
    print(f"   Primary: {nodes['node0'].get_primary()}")
    print(f"   n = {len(replica_ids)}, f = {f}")
    print(f"   Can tolerate up to {f} Byzantine failures")
    
    print("\n2. PBFT Configuration:")
    print(f"   View: {nodes['node0'].view}")
    print(f"   Sequence: {nodes['node0'].sequence}")
    print(f"   State: {nodes['node0'].state.value}")
    
    print("\n3. Message types:")
    for msg_type in MessageType:
        print(f"   - {msg_type.value}")
    
    print("\n4. Consensus requires:")
    print(f"   - 2f+1 = {2*f+1} prepare messages")
    print(f"   - 2f+1 = {2*f+1} commit messages")
    
    print("\n" + "="*70)
    print("Note: Full consensus demonstration requires async message passing.")
    print("In production, PBFT operations will be implemented in Rust for performance.")


if __name__ == "__main__":
    _example_usage()
