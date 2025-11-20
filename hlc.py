"""
Hybrid Logical Clock (HLC) Implementation

Hybrid Logical Clocks combine physical time (NTP) with logical counters
to provide a causality-preserving timestamp that is more intuitive than
pure logical clocks.

HLC guarantees:
1. If e1 → e2 (happens-before), then HLC(e1) < HLC(e2)
2. HLC timestamps are close to physical time (within clock drift)
3. Total ordering of all events (unlike vector clocks)

This is essential for distributed systems, especially for:
- Last-Write-Wins CRDTs
- Distributed transaction ordering
- Causal consistency protocols

References:
- "Logical Physical Clocks and Consistent Snapshots in Globally
   Distributed Databases" (Kulkarni et al., 2014)
- CockroachDB's use of HLC
- Apache Cassandra's timestamp handling
"""

import time
from typing import Tuple
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class HLCTimestamp:
    """
    A Hybrid Logical Clock timestamp.
    
    Consists of:
    - physical_time: Physical clock time (milliseconds since epoch)
    - logical_counter: Logical counter for same physical time
    - node_id: Originating node (for tie-breaking)
    """
    physical_time: int  # milliseconds since epoch
    logical_counter: int  # logical counter
    node_id: str  # originating node
    
    def __lt__(self, other: 'HLCTimestamp') -> bool:
        """Compare timestamps (total order)"""
        if self.physical_time != other.physical_time:
            return self.physical_time < other.physical_time
        if self.logical_counter != other.logical_counter:
            return self.logical_counter < other.logical_counter
        return self.node_id < other.node_id
    
    def __le__(self, other: 'HLCTimestamp') -> bool:
        return self < other or self == other
    
    def __gt__(self, other: 'HLCTimestamp') -> bool:
        return not (self <= other)
    
    def __ge__(self, other: 'HLCTimestamp') -> bool:
        return not (self < other)
    
    def __repr__(self) -> str:
        return f"HLC({self.physical_time}, {self.logical_counter}, {self.node_id})"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'physical_time': self.physical_time,
            'logical_counter': self.logical_counter,
            'node_id': self.node_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HLCTimestamp':
        """Create from dictionary"""
        return cls(
            physical_time=data['physical_time'],
            logical_counter=data['logical_counter'],
            node_id=data['node_id']
        )


class HybridLogicalClock:
    """
    Hybrid Logical Clock implementation.
    
    Maintains a monotonically increasing timestamp that combines
    physical time with a logical counter.
    
    Thread-safe implementation suitable for concurrent use.
    """
    
    def __init__(self, node_id: str):
        """
        Initialize HLC for a node.
        
        Args:
            node_id: Unique identifier for this node
        """
        self.node_id = node_id
        self._lock = threading.Lock()
        
        # HLC state
        self._last_physical = 0
        self._logical_counter = 0
        
        # Maximum clock drift we tolerate (in milliseconds)
        self.max_drift_ms = 5000  # 5 seconds
    
    def _get_physical_time(self) -> int:
        """
        Get current physical time in milliseconds.
        
        Returns:
            Current time in milliseconds since epoch
        """
        return int(time.time() * 1000)
    
    def now(self) -> HLCTimestamp:
        """
        Generate a new HLC timestamp for a local event.
        
        Returns:
            A new HLC timestamp
        """
        with self._lock:
            physical = self._get_physical_time()
            
            if physical > self._last_physical:
                # Physical clock advanced, reset logical counter
                self._last_physical = physical
                self._logical_counter = 0
            else:
                # Physical clock hasn't advanced, increment logical counter
                self._logical_counter += 1
            
            return HLCTimestamp(
                physical_time=self._last_physical,
                logical_counter=self._logical_counter,
                node_id=self.node_id
            )
    
    def update(self, remote_timestamp: HLCTimestamp) -> HLCTimestamp:
        """
        Update HLC on receiving a remote timestamp.
        
        This is called when receiving a message with a timestamp.
        It ensures causality is preserved and generates a new timestamp
        for the receive event.
        
        Args:
            remote_timestamp: Timestamp from remote node
            
        Returns:
            New timestamp for this receive event
            
        Raises:
            ValueError: If clock drift exceeds maximum threshold
        """
        with self._lock:
            physical = self._get_physical_time()
            remote_physical = remote_timestamp.physical_time
            
            # Check for excessive clock drift
            drift = abs(physical - remote_physical)
            if drift > self.max_drift_ms:
                raise ValueError(
                    f"Clock drift too large: {drift}ms exceeds {self.max_drift_ms}ms. "
                    f"Physical time: {physical}, Remote: {remote_physical}"
                )
            
            # Update HLC state
            # Take maximum of physical times
            max_physical = max(physical, self._last_physical, remote_physical)
            
            if max_physical > self._last_physical:
                # Physical time advanced
                self._last_physical = max_physical
                self._logical_counter = 0
            elif max_physical == self._last_physical:
                # Same physical time, increment counter
                self._logical_counter += 1
            
            # If remote timestamp's physical time equals our max,
            # we need to ensure our counter is greater
            if remote_physical == max_physical:
                self._logical_counter = max(
                    self._logical_counter,
                    remote_timestamp.logical_counter + 1
                )
            
            return HLCTimestamp(
                physical_time=self._last_physical,
                logical_counter=self._logical_counter,
                node_id=self.node_id
            )
    
    def compare(self, ts1: HLCTimestamp, ts2: HLCTimestamp) -> int:
        """
        Compare two HLC timestamps.
        
        Args:
            ts1: First timestamp
            ts2: Second timestamp
            
        Returns:
            -1 if ts1 < ts2, 0 if equal, 1 if ts1 > ts2
        """
        if ts1 < ts2:
            return -1
        elif ts1 == ts2:
            return 0
        else:
            return 1


# ============================================================================
# HLC-BASED CRDT SUPPORT
# ============================================================================

@dataclass
class LWWRegister:
    """
    Last-Write-Wins Register using HLC.
    
    A CRDT that resolves conflicts using HLC timestamps.
    This is safer than using wall-clock time due to HLC's
    causality guarantees.
    """
    value: any
    timestamp: HLCTimestamp
    
    def merge(self, other: 'LWWRegister') -> 'LWWRegister':
        """
        Merge with another register (take the later write).
        
        Args:
            other: Another LWW register
            
        Returns:
            Merged register with later value
        """
        if self.timestamp >= other.timestamp:
            return self
        else:
            return other
    
    def update(self, value: any, timestamp: HLCTimestamp) -> 'LWWRegister':
        """
        Update register with new value and timestamp.
        
        Args:
            value: New value
            timestamp: Timestamp of the write
            
        Returns:
            Updated register if timestamp is newer, else self
        """
        if timestamp > self.timestamp:
            return LWWRegister(value, timestamp)
        return self


@dataclass
class HLCVersionedValue:
    """
    A value with HLC version for multi-version concurrency control (MVCC).
    
    This can be used for snapshot isolation and time-travel queries.
    """
    value: any
    version: HLCTimestamp
    deleted: bool = False
    
    def is_visible_at(self, timestamp: HLCTimestamp) -> bool:
        """
        Check if this version is visible at a given timestamp.
        
        Args:
            timestamp: Query timestamp
            
        Returns:
            True if this version should be visible
        """
        return self.version <= timestamp and not self.deleted


class HLCVersionStore:
    """
    Multi-version store using HLC for versioning.
    
    Supports:
    - Snapshot reads at any timestamp
    - Time-travel queries
    - Garbage collection of old versions
    """
    
    def __init__(self):
        self.versions: dict[str, list[HLCVersionedValue]] = {}
        self._lock = threading.Lock()
    
    def write(self, key: str, value: any, timestamp: HLCTimestamp) -> None:
        """
        Write a new version of a key.
        
        Args:
            key: The key
            value: The value
            timestamp: HLC timestamp of the write
        """
        with self._lock:
            if key not in self.versions:
                self.versions[key] = []
            
            versioned = HLCVersionedValue(value, timestamp)
            self.versions[key].append(versioned)
            
            # Keep versions sorted by timestamp
            self.versions[key].sort(key=lambda v: v.version)
    
    def read(self, key: str, timestamp: HLCTimestamp) -> any:
        """
        Read the value of a key at a given timestamp.
        
        Args:
            key: The key
            timestamp: HLC timestamp for the read
            
        Returns:
            The value visible at the timestamp, or None if not found
        """
        with self._lock:
            if key not in self.versions:
                return None
            
            # Find the latest version <= timestamp
            visible = None
            for version in self.versions[key]:
                if version.is_visible_at(timestamp):
                    visible = version
                elif version.version > timestamp:
                    break
            
            return visible.value if visible else None
    
    def delete(self, key: str, timestamp: HLCTimestamp) -> None:
        """
        Mark a key as deleted at a timestamp.
        
        Args:
            key: The key
            timestamp: HLC timestamp of the deletion
        """
        with self._lock:
            if key not in self.versions:
                self.versions[key] = []
            
            versioned = HLCVersionedValue(None, timestamp, deleted=True)
            self.versions[key].append(versioned)
            self.versions[key].sort(key=lambda v: v.version)
    
    def gc_before(self, timestamp: HLCTimestamp) -> int:
        """
        Garbage collect versions older than a timestamp.
        
        Keeps at least one version before the timestamp for each key.
        
        Args:
            timestamp: GC cutoff timestamp
            
        Returns:
            Number of versions collected
        """
        collected = 0
        with self._lock:
            for key in list(self.versions.keys()):
                versions = self.versions[key]
                
                # Find last version before timestamp
                keep_from = 0
                for i, version in enumerate(versions):
                    if version.version < timestamp:
                        keep_from = i
                    else:
                        break
                
                # Keep everything from keep_from onwards
                if keep_from > 0:
                    removed = versions[:keep_from]
                    self.versions[key] = versions[keep_from:]
                    collected += len(removed)
                
                # Remove key if no versions left
                if not self.versions[key]:
                    del self.versions[key]
        
        return collected


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def _example_usage():
    """Demonstrate HLC usage"""
    
    print("Hybrid Logical Clock Example\n" + "="*70)
    
    # Create clocks for two nodes
    clock_a = HybridLogicalClock("node_a")
    clock_b = HybridLogicalClock("node_b")
    
    print("\n1. Local events:")
    ts1 = clock_a.now()
    print(f"   Node A event 1: {ts1}")
    
    ts2 = clock_a.now()
    print(f"   Node A event 2: {ts2}")
    print(f"   ts1 < ts2: {ts1 < ts2}")
    
    print("\n2. Message passing (A → B):")
    ts3 = clock_b.update(ts2)
    print(f"   Node B receives from A: {ts3}")
    print(f"   ts2 < ts3: {ts2 < ts3}")
    
    print("\n3. LWW Register with HLC:")
    reg1 = LWWRegister("value1", ts1)
    reg2 = LWWRegister("value2", ts2)
    merged = reg1.merge(reg2)
    print(f"   Register 1: {reg1}")
    print(f"   Register 2: {reg2}")
    print(f"   Merged: {merged}")
    
    print("\n4. Version Store:")
    store = HLCVersionStore()
    store.write("x", 10, ts1)
    store.write("x", 20, ts2)
    store.write("x", 30, ts3)
    
    print(f"   Value at {ts1}: {store.read('x', ts1)}")
    print(f"   Value at {ts2}: {store.read('x', ts2)}")
    print(f"   Value at {ts3}: {store.read('x', ts3)}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    _example_usage()
