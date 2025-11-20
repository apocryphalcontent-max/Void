"""
Gossip Protocol with Plumtree Optimization

Implements epidemic broadcast with Plumtree (Epidemic Broadcast Trees)
optimization for efficient message dissemination in distributed systems.

Plumtree combines:
- Tree-based broadcast for efficiency (low redundancy)
- Gossip fallback for reliability (healing broken links)

References:
- "Epidemic Broadcast Trees" (Leitão, Pereira, Rodrigues, 2007)
- "HyParView: a membership protocol for reliable gossip-based broadcast"
  (Leitão, Pereira, Rodrigues, 2007)
"""

import time
import random
import hashlib
from typing import Dict, Set, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading


class MessageType(Enum):
    """Gossip message types"""
    GOSSIP = "gossip"  # Full message
    IHAVE = "ihave"    # Message announcement (hash only)
    GRAFT = "graft"    # Request to join tree
    PRUNE = "prune"    # Request to leave tree


@dataclass
class GossipMessage:
    """A gossip protocol message"""
    msg_id: str
    payload: Any
    sender: str
    timestamp: float = field(default_factory=time.time)
    hop_count: int = 0
    
    def __hash__(self):
        return hash(self.msg_id)


@dataclass
class PlumtreeNode:
    """
    A node in the Plumtree gossip protocol.
    
    Maintains two overlay networks:
    - Eager push tree: Tree overlay for efficient broadcast
    - Lazy push network: Backup links for healing
    
    **Complexity:** O(n) message overhead per broadcast (optimal)
    **Reliability:** Self-healing via lazy push
    **Latency:** O(log n) hops (tree depth)
    """
    
    def __init__(self, node_id: str, fanout: int = 3, lazy_threshold: int = 2):
        """
        Initialize Plumtree node.
        
        Args:
            node_id: Unique node identifier
            fanout: Number of eager push peers (tree degree)
            lazy_threshold: Missed messages before GRAFT
        """
        self.node_id = node_id
        self.fanout = fanout
        self.lazy_threshold = lazy_threshold
        
        # Overlay networks
        self.eager_push_peers: Set[str] = set()  # Tree edges
        self.lazy_push_peers: Set[str] = set()   # Backup edges
        
        # Message tracking
        self.received_messages: Dict[str, GossipMessage] = {}
        self.missing_messages: Dict[str, int] = defaultdict(int)  # msg_id -> miss count
        
        # Optimization: Track who we received each message from
        self.message_source: Dict[str, str] = {}  # msg_id -> peer_id
        
        self.lock = threading.Lock()
        
        # Callback for sending messages (set by application)
        self.send_callback: Optional[Callable] = None
    
    def add_peer(self, peer_id: str, eager: bool = True) -> None:
        """
        Add a peer to the overlay.
        
        Args:
            peer_id: Peer node ID
            eager: If True, add to eager push tree; else lazy push
        """
        with self.lock:
            if eager:
                self.eager_push_peers.add(peer_id)
                self.lazy_push_peers.discard(peer_id)
            else:
                self.lazy_push_peers.add(peer_id)
                self.eager_push_peers.discard(peer_id)
    
    def remove_peer(self, peer_id: str) -> None:
        """Remove peer from overlay"""
        with self.lock:
            self.eager_push_peers.discard(peer_id)
            self.lazy_push_peers.discard(peer_id)
    
    def broadcast(self, payload: Any) -> str:
        """
        Broadcast a new message.
        
        Args:
            payload: Message content
            
        Returns:
            Message ID
        """
        # Create message with unique ID
        msg_id = hashlib.sha256(
            f"{self.node_id}{time.time()}{payload}".encode()
        ).hexdigest()[:16]
        
        message = GossipMessage(
            msg_id=msg_id,
            payload=payload,
            sender=self.node_id,
            timestamp=time.time(),
            hop_count=0
        )
        
        with self.lock:
            self.received_messages[msg_id] = message
        
        # Eager push to tree neighbors
        self._eager_push(message)
        
        # Lazy push to backup neighbors (just announcement)
        self._lazy_push(message)
        
        return msg_id
    
    def receive_gossip(self, message: GossipMessage, from_peer: str) -> None:
        """
        Handle received GOSSIP message (full message).
        
        Args:
            message: The gossip message
            from_peer: Peer who sent it
        """
        with self.lock:
            # Check if we've already seen this message
            if message.msg_id in self.received_messages:
                # Duplicate - optimize tree structure
                if from_peer in self.eager_push_peers:
                    # Move peer to lazy push (prune redundant edge)
                    self._send_prune(from_peer, message.msg_id)
                    self.eager_push_peers.remove(from_peer)
                    self.lazy_push_peers.add(from_peer)
                return
            
            # New message - store it
            self.received_messages[message.msg_id] = message
            self.message_source[message.msg_id] = from_peer
            
            # Clear any missing message tracking
            if message.msg_id in self.missing_messages:
                del self.missing_messages[message.msg_id]
        
        # Forward to others
        message.hop_count += 1
        self._eager_push(message, exclude=from_peer)
        self._lazy_push(message, exclude=from_peer)
    
    def receive_ihave(self, msg_id: str, from_peer: str) -> None:
        """
        Handle received IHAVE message (lazy announcement).
        
        Args:
            msg_id: Message identifier
            from_peer: Peer who sent it
        """
        with self.lock:
            if msg_id in self.received_messages:
                # Already have it - no action needed
                return
            
            # Track missing message
            self.missing_messages[msg_id] += 1
            
            # If we've missed too many messages, request to join tree
            if self.missing_messages[msg_id] >= self.lazy_threshold:
                self._send_graft(from_peer, msg_id)
                
                # Move peer from lazy to eager
                if from_peer in self.lazy_push_peers:
                    self.lazy_push_peers.remove(from_peer)
                    self.eager_push_peers.add(from_peer)
    
    def receive_graft(self, peer_id: str, msg_id: str) -> None:
        """
        Handle GRAFT message (peer wants to join tree).
        
        Args:
            peer_id: Peer requesting graft
            msg_id: Message that triggered the graft
        """
        with self.lock:
            # Move peer to eager push
            if peer_id in self.lazy_push_peers:
                self.lazy_push_peers.remove(peer_id)
            self.eager_push_peers.add(peer_id)
            
            # Send the message they missed
            if msg_id in self.received_messages:
                self._send_gossip(peer_id, self.received_messages[msg_id])
    
    def receive_prune(self, peer_id: str, msg_id: str) -> None:
        """
        Handle PRUNE message (peer wants to leave tree).
        
        Args:
            peer_id: Peer requesting prune
            msg_id: Message that triggered the prune
        """
        with self.lock:
            # Move peer to lazy push
            if peer_id in self.eager_push_peers:
                self.eager_push_peers.remove(peer_id)
            self.lazy_push_peers.add(peer_id)
    
    def _eager_push(self, message: GossipMessage, exclude: Optional[str] = None) -> None:
        """
        Eager push: Send full message to tree neighbors.
        
        Args:
            message: Message to send
            exclude: Peer to exclude (typically the sender)
        """
        if not self.send_callback:
            return
        
        peers = self.eager_push_peers.copy()
        if exclude:
            peers.discard(exclude)
        
        for peer_id in peers:
            self._send_gossip(peer_id, message)
    
    def _lazy_push(self, message: GossipMessage, exclude: Optional[str] = None) -> None:
        """
        Lazy push: Send only message ID to backup neighbors.
        
        Args:
            message: Message (only ID is sent)
            exclude: Peer to exclude
        """
        if not self.send_callback:
            return
        
        peers = self.lazy_push_peers.copy()
        if exclude:
            peers.discard(exclude)
        
        for peer_id in peers:
            self._send_ihave(peer_id, message.msg_id)
    
    def _send_gossip(self, peer_id: str, message: GossipMessage) -> None:
        """Send GOSSIP message"""
        if self.send_callback:
            self.send_callback(peer_id, MessageType.GOSSIP, {
                'message': message
            })
    
    def _send_ihave(self, peer_id: str, msg_id: str) -> None:
        """Send IHAVE message"""
        if self.send_callback:
            self.send_callback(peer_id, MessageType.IHAVE, {
                'msg_id': msg_id
            })
    
    def _send_graft(self, peer_id: str, msg_id: str) -> None:
        """Send GRAFT message"""
        if self.send_callback:
            self.send_callback(peer_id, MessageType.GRAFT, {
                'msg_id': msg_id
            })
    
    def _send_prune(self, peer_id: str, msg_id: str) -> None:
        """Send PRUNE message"""
        if self.send_callback:
            self.send_callback(peer_id, MessageType.PRUNE, {
                'msg_id': msg_id
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics"""
        with self.lock:
            return {
                'node_id': self.node_id,
                'eager_peers': len(self.eager_push_peers),
                'lazy_peers': len(self.lazy_push_peers),
                'received_messages': len(self.received_messages),
                'missing_messages': len(self.missing_messages)
            }


class PlumtreeNetwork:
    """
    Simulated Plumtree network for testing.
    
    Connects multiple PlumtreeNode instances.
    """
    
    def __init__(self):
        self.nodes: Dict[str, PlumtreeNode] = {}
        self.message_count = 0
        self.lock = threading.Lock()
    
    def add_node(self, node: PlumtreeNode) -> None:
        """Add node to network"""
        self.nodes[node.node_id] = node
        
        # Set up message callback
        node.send_callback = self._deliver_message
    
    def connect_nodes(self, node1_id: str, node2_id: str, bidirectional: bool = True) -> None:
        """Connect two nodes (add to eager push tree)"""
        if node1_id in self.nodes:
            self.nodes[node1_id].add_peer(node2_id, eager=True)
        
        if bidirectional and node2_id in self.nodes:
            self.nodes[node2_id].add_peer(node1_id, eager=True)
    
    def _deliver_message(self, target_id: str, msg_type: MessageType, data: Dict) -> None:
        """Deliver message to target node"""
        with self.lock:
            self.message_count += 1
        
        if target_id not in self.nodes:
            return
        
        target = self.nodes[target_id]
        
        if msg_type == MessageType.GOSSIP:
            target.receive_gossip(data['message'], data.get('from', 'unknown'))
        elif msg_type == MessageType.IHAVE:
            target.receive_ihave(data['msg_id'], data.get('from', 'unknown'))
        elif msg_type == MessageType.GRAFT:
            target.receive_graft(data.get('from', 'unknown'), data['msg_id'])
        elif msg_type == MessageType.PRUNE:
            target.receive_prune(data.get('from', 'unknown'), data['msg_id'])
    
    def get_total_messages(self) -> int:
        """Get total messages sent"""
        return self.message_count


# Example usage
if __name__ == "__main__":
    print("=== Plumtree Gossip Protocol Example ===\n")
    
    # Create network
    network = PlumtreeNetwork()
    
    # Create nodes
    node_ids = [f"node{i}" for i in range(5)]
    for node_id in node_ids:
        node = PlumtreeNode(node_id, fanout=2)
        network.add_node(node)
    
    # Create tree topology: 0 -> 1, 2 and 1 -> 3, 4
    network.connect_nodes("node0", "node1")
    network.connect_nodes("node0", "node2")
    network.connect_nodes("node1", "node3")
    network.connect_nodes("node1", "node4")
    
    # Add lazy push links for redundancy
    network.nodes["node0"].add_peer("node3", eager=False)
    network.nodes["node0"].add_peer("node4", eager=False)
    
    print("Network topology created:")
    for node_id, node in network.nodes.items():
        stats = node.get_stats()
        print(f"  {node_id}: {stats['eager_peers']} eager, {stats['lazy_peers']} lazy peers")
    
    print("\nBroadcasting message from node0...")
    msg_id = network.nodes["node0"].broadcast({"type": "update", "value": 42})
    
    print(f"Message {msg_id} broadcast")
    print(f"Total messages sent: {network.get_total_messages()}")
    
    # Check message delivery
    print("\nMessage delivery status:")
    for node_id, node in network.nodes.items():
        has_message = msg_id in node.received_messages
        print(f"  {node_id}: {'✓' if has_message else '✗'}")
