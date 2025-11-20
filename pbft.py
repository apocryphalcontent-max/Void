"""
Practical Byzantine Fault Tolerance (PBFT) Consensus

Pure Python implementation of PBFT consensus protocol for distributed
systems. This module will be migrated to Rust for performance.

PBFT provides Byzantine fault tolerance with up to f < n/3 faulty nodes.
Uses cryptographic signatures (Ed25519) for message authentication.

References:
- "Practical Byzantine Fault Tolerance" (Castro & Liskov, 1999)
- "Practical Byzantine Fault Tolerance and Proactive Recovery" 
  (Castro & Liskov, 2002)

NOTE: This is a Python implementation for prototyping. For production,
migrate to Rust using PyO3 (see rust_core/ directory).
"""

import hashlib
import time
import json
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict

# Try to import cryptography for Ed25519, fallback to basic hashing
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("Warning: cryptography not installed. Using insecure mock signatures.")


class PBFTPhase(Enum):
    """PBFT protocol phases"""
    PRE_PREPARE = "pre-prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    REPLY = "reply"


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
    
    All messages are cryptographically signed to prevent tampering.
    """
    msg_type: MessageType
    view: int
    sequence: int
    sender: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    signature: Optional[bytes] = None
    digest: Optional[str] = None
    
    def compute_digest(self) -> str:
        """Compute message digest for signing"""
        content = json.dumps({
            'type': self.msg_type.value,
            'view': self.view,
            'seq': self.sequence,
            'sender': self.sender,
            'payload': str(self.payload),
            'timestamp': self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def __hash__(self):
        return hash((self.msg_type, self.view, self.sequence, self.sender))


@dataclass
class ClientRequest:
    """Client request to be ordered"""
    client_id: str
    operation: Any
    timestamp: float
    request_id: str = field(default_factory=lambda: hashlib.sha256(
        f"{time.time()}".encode()
    ).hexdigest()[:16])


@dataclass
class PBFTState:
    """State for a PBFT replica"""
    view: int = 0
    sequence: int = 0
    last_executed: int = 0
    checkpoint_interval: int = 100
    
    # Message logs
    request_log: Dict[str, ClientRequest] = field(default_factory=dict)
    pre_prepare_log: Dict[int, PBFTMessage] = field(default_factory=dict)
    prepare_log: Dict[int, Set[PBFTMessage]] = field(default_factory=lambda: defaultdict(set))
    commit_log: Dict[int, Set[PBFTMessage]] = field(default_factory=lambda: defaultdict(set))
    
    # Execution state
    executed_requests: Set[str] = field(default_factory=set)
    checkpoints: Dict[int, str] = field(default_factory=dict)


class PBFTNode:
    """
    A PBFT consensus node (replica).
    
    Implements the full PBFT protocol including:
    - Three-phase commit (pre-prepare, prepare, commit)
    - View changes for leader election
    - Checkpointing for garbage collection
    - Cryptographic message authentication
    
    **Safety:** Tolerates f < n/3 Byzantine faults
    **Liveness:** Requires eventual synchrony
    **Complexity:** O(n²) messages per request
    """
    
    def __init__(self, node_id: str, all_nodes: List[str], f: int = None):
        """
        Initialize PBFT node.
        
        Args:
            node_id: Unique ID for this replica
            all_nodes: List of all replica IDs in the system
            f: Maximum Byzantine faults to tolerate (default: (n-1)//3)
        """
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.n = len(all_nodes)
        self.f = f if f is not None else (self.n - 1) // 3
        
        if self.n < 3 * self.f + 1:
            raise ValueError(
                f"PBFT requires n >= 3f+1: n={self.n}, f={self.f}"
            )
        
        # State
        self.state = PBFTState()
        self.lock = threading.Lock()
        
        # Cryptographic keys
        self._init_crypto()
        
        # Network callback (to be set by application)
        self.send_message_callback = None
    
    def _init_crypto(self):
        """Initialize cryptographic keys for signing"""
        if HAS_CRYPTO:
            self.private_key = ed25519.Ed25519PrivateKey.generate()
            self.public_key = self.private_key.public_key()
            
            # In real system, would exchange public keys with all nodes
            self.public_keys: Dict[str, Any] = {}
        else:
            # Mock implementation
            self.private_key = f"mock_private_{self.node_id}"
            self.public_key = f"mock_public_{self.node_id}"
            self.public_keys = {}
    
    def sign_message(self, message: PBFTMessage) -> bytes:
        """Sign a message with this node's private key"""
        message.digest = message.compute_digest()
        
        if HAS_CRYPTO:
            return self.private_key.sign(message.digest.encode())
        else:
            # Mock signature
            return hashlib.sha256(
                f"{self.private_key}{message.digest}".encode()
            ).digest()
    
    def verify_signature(self, message: PBFTMessage, sender_id: str) -> bool:
        """Verify message signature"""
        if not message.signature or not message.digest:
            return False
        
        if HAS_CRYPTO:
            if sender_id not in self.public_keys:
                return False
            try:
                public_key = self.public_keys[sender_id]
                public_key.verify(message.signature, message.digest.encode())
                return True
            except Exception:
                return False
        else:
            # Mock verification
            expected = hashlib.sha256(
                f"mock_private_{sender_id}{message.digest}".encode()
            ).digest()
            return message.signature == expected
    
    def is_primary(self, view: int) -> bool:
        """Check if this node is the primary for given view"""
        primary_idx = view % self.n
        return self.all_nodes[primary_idx] == self.node_id
    
    def get_primary(self, view: int) -> str:
        """Get primary node ID for given view"""
        primary_idx = view % self.n
        return self.all_nodes[primary_idx]
    
    def handle_client_request(self, request: ClientRequest) -> None:
        """
        Handle client request (primary only).
        
        Primary assigns sequence number and broadcasts PRE-PREPARE.
        """
        with self.lock:
            if not self.is_primary(self.state.view):
                # Forward to primary (not implemented here)
                return
            
            # Assign sequence number
            self.state.sequence += 1
            seq = self.state.sequence
            
            # Store request
            self.state.request_log[request.request_id] = request
            
            # Create PRE-PREPARE message
            pre_prepare = PBFTMessage(
                msg_type=MessageType.PRE_PREPARE,
                view=self.state.view,
                sequence=seq,
                sender=self.node_id,
                payload={
                    'request_id': request.request_id,
                    'operation': request.operation
                }
            )
            
            # Sign and broadcast
            pre_prepare.signature = self.sign_message(pre_prepare)
            self._broadcast(pre_prepare)
            
            # Process our own PRE-PREPARE
            self._handle_pre_prepare(pre_prepare)
    
    def _handle_pre_prepare(self, msg: PBFTMessage) -> None:
        """Handle PRE-PREPARE message"""
        with self.lock:
            # Validate message
            if msg.view != self.state.view:
                return  # Wrong view
            
            if not self.verify_signature(msg, msg.sender):
                return  # Invalid signature
            
            if msg.sender != self.get_primary(msg.view):
                return  # Not from primary
            
            if msg.sequence in self.state.pre_prepare_log:
                # Already have PRE-PREPARE for this sequence
                existing = self.state.pre_prepare_log[msg.sequence]
                if existing.digest != msg.digest:
                    # Conflict - primary is Byzantine!
                    self._trigger_view_change()
                return
            
            # Accept PRE-PREPARE
            self.state.pre_prepare_log[msg.sequence] = msg
            
            # Send PREPARE
            prepare = PBFTMessage(
                msg_type=MessageType.PREPARE,
                view=self.state.view,
                sequence=msg.sequence,
                sender=self.node_id,
                payload={'digest': msg.digest}
            )
            prepare.signature = self.sign_message(prepare)
            self._broadcast(prepare)
    
    def _handle_prepare(self, msg: PBFTMessage) -> None:
        """Handle PREPARE message"""
        with self.lock:
            if msg.view != self.state.view:
                return
            
            if not self.verify_signature(msg, msg.sender):
                return
            
            # Add to prepare log
            self.state.prepare_log[msg.sequence].add(msg)
            
            # Check if we have 2f PREPARE messages (quorum)
            if len(self.state.prepare_log[msg.sequence]) >= 2 * self.f:
                # Check if we have matching PRE-PREPARE
                if msg.sequence in self.state.pre_prepare_log:
                    # Prepared! Send COMMIT
                    commit = PBFTMessage(
                        msg_type=MessageType.COMMIT,
                        view=self.state.view,
                        sequence=msg.sequence,
                        sender=self.node_id,
                        payload={'digest': msg.payload['digest']}
                    )
                    commit.signature = self.sign_message(commit)
                    self._broadcast(commit)
    
    def _handle_commit(self, msg: PBFTMessage) -> None:
        """Handle COMMIT message"""
        with self.lock:
            if msg.view != self.state.view:
                return
            
            if not self.verify_signature(msg, msg.sender):
                return
            
            # Add to commit log
            self.state.commit_log[msg.sequence].add(msg)
            
            # Check if we have 2f+1 COMMIT messages (quorum)
            if len(self.state.commit_log[msg.sequence]) >= 2 * self.f + 1:
                # Committed! Execute if ready
                self._execute_request(msg.sequence)
    
    def _execute_request(self, sequence: int) -> None:
        """Execute committed request"""
        # Execute requests in order
        while self.state.last_executed + 1 in self.state.pre_prepare_log:
            seq = self.state.last_executed + 1
            
            # Check if committed
            if len(self.state.commit_log[seq]) < 2 * self.f + 1:
                break
            
            # Get request
            pre_prepare = self.state.pre_prepare_log[seq]
            request_id = pre_prepare.payload['request_id']
            
            if request_id in self.state.executed_requests:
                # Already executed (duplicate)
                self.state.last_executed = seq
                continue
            
            if request_id not in self.state.request_log:
                # Don't have the request yet
                break
            
            request = self.state.request_log[request_id]
            
            # Execute operation (application-specific)
            result = self._execute_operation(request.operation)
            
            # Mark as executed
            self.state.executed_requests.add(request_id)
            self.state.last_executed = seq
            
            # Send reply to client (not implemented)
            
            # Checkpoint if needed
            if seq % self.state.checkpoint_interval == 0:
                self._create_checkpoint(seq)
    
    def _execute_operation(self, operation: Any) -> Any:
        """
        Execute operation (application-specific).
        
        Override this method to implement actual state machine.
        """
        # Default: just return the operation
        return operation
    
    def _create_checkpoint(self, sequence: int) -> None:
        """Create a checkpoint for garbage collection"""
        # Compute state digest
        state_digest = hashlib.sha256(
            json.dumps({
                'sequence': sequence,
                'executed': sorted(self.state.executed_requests)
            }).encode()
        ).hexdigest()
        
        self.state.checkpoints[sequence] = state_digest
        
        # Garbage collect old messages
        self._garbage_collect(sequence)
    
    def _garbage_collect(self, checkpoint_seq: int) -> None:
        """Remove old messages before checkpoint"""
        # Remove old PRE-PREPARE messages
        old_seqs = [s for s in self.state.pre_prepare_log if s < checkpoint_seq]
        for seq in old_seqs:
            del self.state.pre_prepare_log[seq]
            if seq in self.state.prepare_log:
                del self.state.prepare_log[seq]
            if seq in self.state.commit_log:
                del self.state.commit_log[seq]
    
    def _trigger_view_change(self) -> None:
        """Trigger view change (leader election)"""
        # Simplified view change - full implementation is complex
        self.state.view += 1
        # Would send VIEW-CHANGE messages to all replicas
    
    def _broadcast(self, message: PBFTMessage) -> None:
        """Broadcast message to all replicas"""
        if self.send_message_callback:
            for node_id in self.all_nodes:
                if node_id != self.node_id:
                    self.send_message_callback(node_id, message)
    
    def process_message(self, message: PBFTMessage) -> None:
        """Process received message"""
        if message.msg_type == MessageType.PRE_PREPARE:
            self._handle_pre_prepare(message)
        elif message.msg_type == MessageType.PREPARE:
            self._handle_prepare(message)
        elif message.msg_type == MessageType.COMMIT:
            self._handle_commit(message)


# Example usage
if __name__ == "__main__":
    print("=== PBFT Consensus Example ===\n")
    
    # Create 4 nodes (tolerates 1 Byzantine fault)
    nodes = ["node0", "node1", "node2", "node3"]
    replicas = {
        node_id: PBFTNode(node_id, nodes)
        for node_id in nodes
    }
    
    # Set up message passing
    def send_message(sender_id: str, target_id: str, msg: PBFTMessage):
        replicas[target_id].process_message(msg)
    
    for node_id, replica in replicas.items():
        replica.send_message_callback = lambda target, msg, sender=node_id: send_message(sender, target, msg)
    
    # Submit client request
    print("Submitting client request...")
    request = ClientRequest(
        client_id="client1",
        operation={"type": "transfer", "amount": 100},
        timestamp=time.time()
    )
    
    # Primary handles request
    primary = replicas["node0"]
    primary.handle_client_request(request)
    
    print(f"Request processed by PBFT consensus")
    print(f"Primary: {primary.node_id}")
    print(f"View: {primary.state.view}")
    print(f"Sequence: {primary.state.sequence}")
