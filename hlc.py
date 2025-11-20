"""
Hybrid Logical Clocks (HLC) for Void-State System

Implements Hybrid Logical Clocks that combine physical time (NTP) with
logical counters to provide causal ordering without drift issues.

Critical for CRDTs and distributed consensus where wall-clock time
can cause inconsistencies.

References:
- "Logical Physical Clocks and Consistent Snapshots in Globally 
  Distributed Databases" (Kulkarni & Demirbas, 2014)
- CockroachDB's HLC implementation
- TiKV's timestamp oracle
"""

import time
import threading
from dataclasses import dataclass
from typing import Tuple, Optional
import struct


@dataclass(frozen=True)
class HLCTimestamp:
    """
    A Hybrid Logical Clock timestamp.
    
    Combines:
    - physical_time: Wall clock time (microseconds since epoch)
    - logical: Logical counter for events with same physical time
    
    Ordering: (t1.physical, t1.logical) < (t2.physical, t2.logical)
    """
    physical_time: int  # microseconds since epoch
    logical: int  # logical counter
    
    def __lt__(self, other: 'HLCTimestamp') -> bool:
        """Compare timestamps using hybrid ordering"""
        if self.physical_time != other.physical_time:
            return self.physical_time < other.physical_time
        return self.logical < other.logical
    
    def __le__(self, other: 'HLCTimestamp') -> bool:
        return self < other or self == other
    
    def __gt__(self, other: 'HLCTimestamp') -> bool:
        return other < self
    
    def __ge__(self, other: 'HLCTimestamp') -> bool:
        return other <= self
    
    def to_bytes(self) -> bytes:
        """Serialize timestamp to bytes"""
        return struct.pack('>QI', self.physical_time, self.logical)
    
    @staticmethod
    def from_bytes(data: bytes) -> 'HLCTimestamp':
        """Deserialize timestamp from bytes"""
        physical, logical = struct.unpack('>QI', data)
        return HLCTimestamp(physical, logical)
    
    def __repr__(self) -> str:
        return f"HLC({self.physical_time}μs, l={self.logical})"


class HybridLogicalClock:
    """
    Hybrid Logical Clock implementation.
    
    Thread-safe clock that maintains causality while tracking physical time.
    Used for ordering events in distributed systems without synchronization.
    
    Properties:
    1. If event e1 happens-before e2 on same node: HLC(e1) < HLC(e2)
    2. If event e1 on node A sends message to event e2 on node B: HLC(e1) < HLC(e2)
    3. Physical time is close to wall clock (bounded drift)
    """
    
    def __init__(self, node_id: str, max_drift_ms: int = 500):
        """
        Initialize HLC.
        
        Args:
            node_id: Unique identifier for this node
            max_drift_ms: Maximum allowed drift from physical time (milliseconds)
        """
        self.node_id = node_id
        self.max_drift_us = max_drift_ms * 1000  # Convert to microseconds
        
        # State (protected by lock)
        self._lock = threading.Lock()
        self._last_timestamp = HLCTimestamp(0, 0)
    
    def now(self) -> HLCTimestamp:
        """
        Generate a new timestamp for a local event.
        
        Returns:
            A new HLC timestamp that preserves causality
        """
        with self._lock:
            physical_now = self._get_physical_time()
            
            if physical_now > self._last_timestamp.physical_time:
                # Physical time advanced - use it with logical = 0
                new_timestamp = HLCTimestamp(physical_now, 0)
            else:
                # Physical time hasn't advanced or went backwards
                # Increment logical counter
                new_timestamp = HLCTimestamp(
                    self._last_timestamp.physical_time,
                    self._last_timestamp.logical + 1
                )
            
            # Check for drift
            drift = abs(new_timestamp.physical_time - physical_now)
            if drift > self.max_drift_us:
                # Clock has drifted too far from physical time
                # This can happen if logical counter keeps incrementing
                # without physical time advancing
                raise ClockDriftError(
                    f"HLC drifted {drift/1000:.2f}ms from physical time "
                    f"(max: {self.max_drift_us/1000:.2f}ms). "
                    f"This indicates either clock skew or too many events."
                )
            
            self._last_timestamp = new_timestamp
            return new_timestamp
    
    def update(self, received_timestamp: HLCTimestamp) -> HLCTimestamp:
        """
        Update clock based on received message timestamp.
        
        This is called when receiving a message from another node.
        Ensures the returned timestamp is greater than both the local
        time and the received timestamp.
        
        Args:
            received_timestamp: Timestamp from received message
            
        Returns:
            Updated HLC timestamp for the receive event
        """
        with self._lock:
            physical_now = self._get_physical_time()
            
            # Take the maximum of local time and received time
            max_physical = max(
                physical_now,
                self._last_timestamp.physical_time,
                received_timestamp.physical_time
            )
            
            # Determine logical counter
            if max_physical == physical_now and max_physical > self._last_timestamp.physical_time and max_physical > received_timestamp.physical_time:
                # Physical time advanced past both clocks
                logical = 0
            elif max_physical == self._last_timestamp.physical_time and max_physical > received_timestamp.physical_time:
                # Our physical time is ahead
                logical = self._last_timestamp.logical + 1
            elif max_physical == received_timestamp.physical_time and max_physical > self._last_timestamp.physical_time:
                # Received physical time is ahead
                logical = received_timestamp.logical + 1
            else:
                # All three times are equal
                logical = max(
                    self._last_timestamp.logical,
                    received_timestamp.logical
                ) + 1
            
            new_timestamp = HLCTimestamp(max_physical, logical)
            
            # Check drift
            drift = abs(new_timestamp.physical_time - physical_now)
            if drift > self.max_drift_us:
                raise ClockDriftError(
                    f"HLC would drift {drift/1000:.2f}ms from physical time "
                    f"after update (max: {self.max_drift_us/1000:.2f}ms)"
                )
            
            self._last_timestamp = new_timestamp
            return new_timestamp
    
    def get_last_timestamp(self) -> HLCTimestamp:
        """Get the last issued timestamp (thread-safe)"""
        with self._lock:
            return self._last_timestamp
    
    @staticmethod
    def _get_physical_time() -> int:
        """Get current physical time in microseconds since epoch"""
        return int(time.time() * 1_000_000)
    
    def __repr__(self) -> str:
        return f"HLC(node={self.node_id}, last={self._last_timestamp})"


class ClockDriftError(Exception):
    """Raised when HLC drifts too far from physical time"""
    pass


def happens_before(ts1: HLCTimestamp, ts2: HLCTimestamp) -> bool:
    """Check if ts1 happens-before ts2"""
    return ts1 < ts2


def concurrent(ts1: HLCTimestamp, ts2: HLCTimestamp) -> bool:
    """Check if two timestamps are concurrent (causally independent)"""
    return ts1 != ts2 and not (ts1 < ts2) and not (ts2 < ts1)


# Example usage and tests
if __name__ == "__main__":
    print("=== Hybrid Logical Clock Examples ===\n")
    
    # Example 1: Basic local events
    print("Example 1: Local events on single node")
    clock_a = HybridLogicalClock("node_a")
    
    t1 = clock_a.now()
    print(f"Event 1: {t1}")
    
    time.sleep(0.001)  # 1ms delay
    
    t2 = clock_a.now()
    print(f"Event 2: {t2}")
    print(f"t1 < t2: {t1 < t2}")
    print(f"t1 happens-before t2: {happens_before(t1, t2)}\n")
    
    # Example 2: Message passing between nodes
    print("Example 2: Message passing")
    clock_a = HybridLogicalClock("node_a")
    clock_b = HybridLogicalClock("node_b")
    
    # Node A sends message
    t_send = clock_a.now()
    print(f"Node A sends at: {t_send}")
    
    # Simulate network delay
    time.sleep(0.002)
    
    # Node B receives message
    t_receive = clock_b.update(t_send)
    print(f"Node B receives at: {t_receive}")
    print(f"Causality preserved: {t_send < t_receive}\n")
    
    # Example 3: Rapid events (logical counter increments)
    print("Example 3: Rapid events (tests logical counter)")
    clock_c = HybridLogicalClock("node_c")
    
    timestamps = []
    for i in range(5):
        ts = clock_c.now()
        timestamps.append(ts)
        print(f"Event {i}: {ts}")
    
    # Verify total ordering
    print("All timestamps ordered:", all(
        timestamps[i] < timestamps[i+1]
        for i in range(len(timestamps)-1)
    ))
    print()
    
    # Example 4: Serialization
    print("Example 4: Timestamp serialization")
    ts = clock_a.now()
    print(f"Original: {ts}")
    
    serialized = ts.to_bytes()
    print(f"Serialized: {serialized.hex()}")
    
    deserialized = HLCTimestamp.from_bytes(serialized)
    print(f"Deserialized: {deserialized}")
    print(f"Equal: {ts == deserialized}")
