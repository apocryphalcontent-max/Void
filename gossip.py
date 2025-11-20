"""
Gossip Protocol with Plumtree Optimization

Implements epidemic-style gossip protocols for distributed message dissemination,
optimized using Plumtree (epidemic broadcast trees) for efficiency.

Plumtree combines:
1. Eager push for fast dissemination (spanning tree)
2. Lazy push for reliability (gossip fallback)
3. Self-healing when tree breaks

This reduces message overhead from O(n²) to O(n) while maintaining
the reliability of gossip.

References:
- "Epidemic Broadcast Trees" (Leitão, Pereira, Rodrigues, 2007)
- "HyParView: A Membership Protocol for Reliable Gossip-Based Broadcast"
- Plumtree implementation in Riak and other systems
"""

from typing import Dict, Set, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import threading
from collections import defaultdict
import random


# ============================================================================
# GOSSIP MESSAGE TYPES
# ============================================================================

class MessageType(Enum):
    """Types of gossip messages"""
    GOSSIP = "gossip"  # Full message (eager push)
    IHAVE = "ihave"    # Announcement (lazy push)
    GRAFT = "graft"    # Request to join eager push
    PRUNE = "prune"    # Leave eager push tree


@dataclass
class GossipMessage:
    """
    A gossip message.
    
    Contains:
    - msg_id: Unique message identifier
    - sender: Sending node
    - payload: Actual data
    - round: Gossip round number
    - timestamp: Message creation time
    """
    msg_id: str
    sender: str
    payload: Any
    round: int
    timestamp: float
    
    def __hash__(self):
        return hash(self.msg_id)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'msg_id': self.msg_id,
            'sender': self.sender,
            'payload': self.payload,
            'round': self.round,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GossipMessage':
        """Deserialize from dictionary"""
        return cls(
            msg_id=data['msg_id'],
            sender=data['sender'],
            payload=data['payload'],
            round=data['round'],
            timestamp=data['timestamp']
        )


@dataclass
class ControlMessage:
    """Control message for Plumtree protocol"""
    msg_type: MessageType
    msg_id: str
    sender: str
    
    def to_dict(self) -> dict:
        return {
            'msg_type': self.msg_type.value,
            'msg_id': self.msg_id,
            'sender': self.sender
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ControlMessage':
        return cls(
            msg_type=MessageType(data['msg_type']),
            msg_id=data['msg_id'],
            sender=data['sender']
        )


# ============================================================================
# PLUMTREE GOSSIP PROTOCOL
# ============================================================================

class PlumtreeNode:
    """
    Node implementing the Plumtree epidemic broadcast tree protocol.
    
    Maintains two views of neighbors:
    - eager_push: Nodes in the spanning tree (receive full messages)
    - lazy_push: Nodes outside tree (receive only announcements)
    
    When a node receives a duplicate message, it sends PRUNE to optimize
    the tree structure over time.
    """
    
    def __init__(
        self,
        node_id: str,
        max_eager_peers: int = 6,
        max_lazy_peers: int = 12,
        gossip_fanout: int = 3,
        ihave_timeout: float = 1.0
    ):
        """
        Initialize Plumtree node.
        
        Args:
            node_id: Unique node identifier
            max_eager_peers: Maximum eager push peers
            max_lazy_peers: Maximum lazy push peers
            gossip_fanout: Number of random peers for gossip
            ihave_timeout: Timeout for IHAVE messages (seconds)
        """
        self.node_id = node_id
        self.max_eager_peers = max_eager_peers
        self.max_lazy_peers = max_lazy_peers
        self.gossip_fanout = gossip_fanout
        self.ihave_timeout = ihave_timeout
        
        # Peer sets
        self.eager_push_peers: Set[str] = set()
        self.lazy_push_peers: Set[str] = set()
        self.all_peers: Set[str] = set()
        
        # Message tracking
        self.received_messages: Dict[str, GossipMessage] = {}
        self.missing_messages: Dict[str, Set[str]] = defaultdict(set)  # msg_id -> nodes
        self.ihave_timers: Dict[str, float] = {}  # msg_id -> timeout time
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'duplicates': 0,
            'prunes_sent': 0,
            'grafts_sent': 0,
        }
        
        self._lock = threading.Lock()
        
        # Callback for message delivery
        self.on_message: Optional[Callable[[GossipMessage], None]] = None
        
        # Callback for sending messages (must be set by user)
        self.send_to_peer: Optional[Callable[[str, dict], None]] = None
    
    def add_peer(self, peer_id: str, eager: bool = True) -> None:
        """
        Add a peer to the gossip network.
        
        Args:
            peer_id: Peer identifier
            eager: If True, add to eager push set, else lazy push
        """
        with self._lock:
            self.all_peers.add(peer_id)
            
            if eager and len(self.eager_push_peers) < self.max_eager_peers:
                self.eager_push_peers.add(peer_id)
            else:
                self.lazy_push_peers.add(peer_id)
    
    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from all sets"""
        with self._lock:
            self.all_peers.discard(peer_id)
            self.eager_push_peers.discard(peer_id)
            self.lazy_push_peers.discard(peer_id)
    
    def broadcast(self, payload: Any) -> str:
        """
        Broadcast a message to all peers.
        
        Args:
            payload: Message payload
            
        Returns:
            Message ID
        """
        # Create message
        msg_id = self._generate_message_id(payload)
        msg = GossipMessage(
            msg_id=msg_id,
            sender=self.node_id,
            payload=payload,
            round=0,
            timestamp=time.time()
        )
        
        # Store locally
        with self._lock:
            self.received_messages[msg_id] = msg
            self.stats['messages_sent'] += 1
        
        # Deliver locally
        if self.on_message:
            self.on_message(msg)
        
        # Send to peers
        self._disseminate_message(msg)
        
        return msg_id
    
    def receive_gossip(self, msg: GossipMessage, from_peer: str) -> None:
        """
        Receive a full gossip message.
        
        Args:
            msg: The gossip message
            from_peer: Sender node
        """
        with self._lock:
            self.stats['messages_received'] += 1
            
            # Check if we've seen this message
            if msg.msg_id in self.received_messages:
                # Duplicate! Send PRUNE to optimize tree
                self.stats['duplicates'] += 1
                self._send_prune(from_peer, msg.msg_id)
                return
            
            # New message - store and deliver
            self.received_messages[msg.msg_id] = msg
            
            # Remove from missing if we were waiting for it
            if msg.msg_id in self.missing_messages:
                del self.missing_messages[msg.msg_id]
                if msg.msg_id in self.ihave_timers:
                    del self.ihave_timers[msg.msg_id]
        
        # Deliver locally
        if self.on_message:
            self.on_message(msg)
        
        # Forward to neighbors
        self._disseminate_message(msg)
    
    def receive_ihave(self, msg_id: str, from_peer: str) -> None:
        """
        Receive an IHAVE (lazy push) announcement.
        
        Args:
            msg_id: Message identifier
            from_peer: Announcing node
        """
        with self._lock:
            # Ignore if we already have it
            if msg_id in self.received_messages:
                return
            
            # Track that this peer has the message
            self.missing_messages[msg_id].add(from_peer)
            
            # Set timeout to request message if not received
            if msg_id not in self.ihave_timers:
                self.ihave_timers[msg_id] = time.time() + self.ihave_timeout
    
    def receive_graft(self, msg_id: str, from_peer: str) -> None:
        """
        Receive a GRAFT request to join eager push tree.
        
        Args:
            msg_id: Message identifier
            from_peer: Requesting node
        """
        with self._lock:
            # Move peer from lazy to eager
            if from_peer in self.lazy_push_peers:
                self.lazy_push_peers.remove(from_peer)
                self.eager_push_peers.add(from_peer)
            
            # If we have the message, send it
            if msg_id in self.received_messages:
                msg = self.received_messages[msg_id]
                if self.send_to_peer:
                    self.send_to_peer(from_peer, {
                        'type': 'gossip',
                        'message': msg.to_dict()
                    })
    
    def receive_prune(self, msg_id: str, from_peer: str) -> None:
        """
        Receive a PRUNE to leave eager push tree.
        
        Args:
            msg_id: Message identifier
            from_peer: Pruning node
        """
        with self._lock:
            self.stats['prunes_sent'] += 1
            
            # Move peer from eager to lazy
            if from_peer in self.eager_push_peers:
                self.eager_push_peers.remove(from_peer)
                self.lazy_push_peers.add(from_peer)
    
    def tick(self) -> None:
        """
        Periodic maintenance tick.
        
        Should be called regularly to:
        - Request missing messages
        - Clean up old state
        """
        current_time = time.time()
        
        with self._lock:
            # Check for missing messages that timed out
            expired = []
            for msg_id, timeout_time in self.ihave_timers.items():
                if current_time >= timeout_time:
                    expired.append(msg_id)
            
            # Request missing messages
            for msg_id in expired:
                if msg_id in self.missing_messages:
                    peers = self.missing_messages[msg_id]
                    if peers:
                        # Pick random peer and send GRAFT
                        peer = random.choice(list(peers))
                        self._send_graft(peer, msg_id)
                
                del self.ihave_timers[msg_id]
    
    def _disseminate_message(self, msg: GossipMessage) -> None:
        """
        Disseminate a message using hybrid eager/lazy push.
        
        Args:
            msg: Message to disseminate
        """
        if not self.send_to_peer:
            return
        
        with self._lock:
            # Eager push: Send full message to eager peers
            for peer in self.eager_push_peers:
                if peer != msg.sender:  # Don't send back to sender
                    self.send_to_peer(peer, {
                        'type': 'gossip',
                        'message': msg.to_dict()
                    })
            
            # Lazy push: Send IHAVE to lazy peers
            for peer in self.lazy_push_peers:
                if peer != msg.sender:
                    self.send_to_peer(peer, {
                        'type': 'ihave',
                        'msg_id': msg.msg_id,
                        'sender': self.node_id
                    })
    
    def _send_prune(self, peer: str, msg_id: str) -> None:
        """Send PRUNE control message"""
        if self.send_to_peer:
            self.send_to_peer(peer, {
                'type': 'prune',
                'msg_id': msg_id,
                'sender': self.node_id
            })
    
    def _send_graft(self, peer: str, msg_id: str) -> None:
        """Send GRAFT control message"""
        with self._lock:
            self.stats['grafts_sent'] += 1
        
        if self.send_to_peer:
            self.send_to_peer(peer, {
                'type': 'graft',
                'msg_id': msg_id,
                'sender': self.node_id
            })
    
    def _generate_message_id(self, payload: Any) -> str:
        """Generate unique message ID"""
        data = f"{self.node_id}:{time.time()}:{payload}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_stats(self) -> dict:
        """Get gossip statistics"""
        with self._lock:
            return {
                **self.stats,
                'eager_peers': len(self.eager_push_peers),
                'lazy_peers': len(self.lazy_push_peers),
                'total_peers': len(self.all_peers),
                'received_messages': len(self.received_messages),
                'missing_messages': len(self.missing_messages),
            }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def _example_usage():
    """Demonstrate Plumtree gossip protocol"""
    
    print("Plumtree Gossip Protocol Example\n" + "="*70)
    
    # Create a small network
    nodes = {
        'A': PlumtreeNode('A'),
        'B': PlumtreeNode('B'),
        'C': PlumtreeNode('C'),
        'D': PlumtreeNode('D'),
    }
    
    # Set up message delivery callbacks
    def make_send_callback(nodes_dict):
        def send_to_peer(sender_id, peer_id, message):
            msg_type = message['type']
            if msg_type == 'gossip':
                msg = GossipMessage.from_dict(message['message'])
                nodes_dict[peer_id].receive_gossip(msg, sender_id)
            elif msg_type == 'ihave':
                nodes_dict[peer_id].receive_ihave(message['msg_id'], sender_id)
            elif msg_type == 'graft':
                nodes_dict[peer_id].receive_graft(message['msg_id'], sender_id)
            elif msg_type == 'prune':
                nodes_dict[peer_id].receive_prune(message['msg_id'], sender_id)
        return send_to_peer
    
    # Configure each node
    for node_id, node in nodes.items():
        sender_id = node_id
        node.send_to_peer = lambda peer, msg, sid=sender_id: make_send_callback(nodes)(sid, peer, msg)
        
        received = []
        node.on_message = lambda msg, rcv=received: rcv.append(msg)
    
    # Create topology: A <-> B <-> C <-> D
    nodes['A'].add_peer('B', eager=True)
    nodes['B'].add_peer('A', eager=True)
    nodes['B'].add_peer('C', eager=True)
    nodes['C'].add_peer('B', eager=True)
    nodes['C'].add_peer('D', eager=True)
    nodes['D'].add_peer('C', eager=True)
    
    # Broadcast a message from A
    print("\n1. Broadcasting message from A:")
    msg_id = nodes['A'].broadcast("Hello from A")
    print(f"   Message ID: {msg_id}")
    
    # Check stats
    print("\n2. Statistics:")
    for node_id, node in nodes.items():
        stats = node.get_stats()
        print(f"   Node {node_id}: {stats}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    _example_usage()
