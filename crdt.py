"""
Conflict-free Replicated Data Types (CRDTs) with HLC

Implements state-based (CvRDT) and operation-based (CmRDT) data structures
that automatically resolve conflicts in distributed systems.

Uses Hybrid Logical Clocks (HLC) for causally-consistent ordering.

References:
- "A comprehensive study of Convergent and Commutative Replicated Data Types"
  (Shapiro et al., 2011)
- "Conflict-free Replicated Data Types" (Shapiro et al., 2011)
- Riak's CRDT implementation
"""

from typing import Any, Dict, Set, List, Optional, Generic, TypeVar
from dataclasses import dataclass, field
from collections import defaultdict
import json

from hlc import HLCTimestamp, HybridLogicalClock


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# ============================================================================
# STATE-BASED CRDTs (CvRDT)
# ============================================================================

class GCounter:
    """
    Grow-only Counter (G-Counter).
    
    A counter that can only increment. Converges to the sum of all increments.
    
    **Merge:** Take maximum of each replica's counter
    **Complexity:** O(n) space for n replicas
    """
    
    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.counts: Dict[str, int] = defaultdict(int)
    
    def increment(self, amount: int = 1) -> None:
        """Increment counter by amount"""
        if amount < 0:
            raise ValueError("G-Counter can only increment (amount must be >= 0)")
        self.counts[self.replica_id] += amount
    
    def value(self) -> int:
        """Get current counter value"""
        return sum(self.counts.values())
    
    def merge(self, other: 'GCounter') -> 'GCounter':
        """Merge with another G-Counter"""
        result = GCounter(self.replica_id)
        all_replicas = set(self.counts.keys()) | set(other.counts.keys())
        
        for replica in all_replicas:
            result.counts[replica] = max(
                self.counts.get(replica, 0),
                other.counts.get(replica, 0)
            )
        
        return result
    
    def __repr__(self) -> str:
        return f"GCounter(value={self.value()})"


class PNCounter:
    """
    Positive-Negative Counter (PN-Counter).
    
    A counter that can both increment and decrement.
    Uses two G-Counters internally.
    
    **Merge:** Merge both underlying G-Counters
    **Value:** P - N (positive minus negative)
    """
    
    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.positive = GCounter(replica_id)
        self.negative = GCounter(replica_id)
    
    def increment(self, amount: int = 1) -> None:
        """Increment counter"""
        if amount < 0:
            raise ValueError("Use decrement() for negative amounts")
        self.positive.increment(amount)
    
    def decrement(self, amount: int = 1) -> None:
        """Decrement counter"""
        if amount < 0:
            raise ValueError("Amount must be positive")
        self.negative.increment(amount)
    
    def value(self) -> int:
        """Get current counter value"""
        return self.positive.value() - self.negative.value()
    
    def merge(self, other: 'PNCounter') -> 'PNCounter':
        """Merge with another PN-Counter"""
        result = PNCounter(self.replica_id)
        result.positive = self.positive.merge(other.positive)
        result.negative = self.negative.merge(other.negative)
        return result
    
    def __repr__(self) -> str:
        return f"PNCounter(value={self.value()})"


class GSet(Generic[T]):
    """
    Grow-only Set (G-Set).
    
    A set that can only add elements, never remove.
    
    **Merge:** Union of both sets
    **Query:** Standard set membership
    """
    
    def __init__(self):
        self.elements: Set[T] = set()
    
    def add(self, element: T) -> None:
        """Add element to set"""
        self.elements.add(element)
    
    def contains(self, element: T) -> bool:
        """Check if element is in set"""
        return element in self.elements
    
    def merge(self, other: 'GSet[T]') -> 'GSet[T]':
        """Merge with another G-Set"""
        result = GSet[T]()
        result.elements = self.elements | other.elements
        return result
    
    def __repr__(self) -> str:
        return f"GSet({self.elements})"


class TwoPhaseSet(Generic[T]):
    """
    Two-Phase Set (2P-Set).
    
    A set that allows adds and removes, but elements cannot be re-added
    after removal (tombstone).
    
    **Merge:** Union of added and removed sets
    **Query:** element in added AND element not in removed
    """
    
    def __init__(self):
        self.added = GSet[T]()
        self.removed = GSet[T]()
    
    def add(self, element: T) -> None:
        """Add element to set"""
        self.added.add(element)
    
    def remove(self, element: T) -> None:
        """Remove element from set"""
        if not self.added.contains(element):
            raise ValueError("Cannot remove element that was never added")
        self.removed.add(element)
    
    def contains(self, element: T) -> bool:
        """Check if element is in set"""
        return self.added.contains(element) and not self.removed.contains(element)
    
    def merge(self, other: 'TwoPhaseSet[T]') -> 'TwoPhaseSet[T]':
        """Merge with another 2P-Set"""
        result = TwoPhaseSet[T]()
        result.added = self.added.merge(other.added)
        result.removed = self.removed.merge(other.removed)
        return result
    
    def __repr__(self) -> str:
        active = {e for e in self.added.elements if not self.removed.contains(e)}
        return f"TwoPhaseSet({active})"


class LWWRegister(Generic[V]):
    """
    Last-Write-Wins Register (LWW-Register).
    
    A register that stores a single value. Conflicts are resolved using
    Hybrid Logical Clock timestamps (last write wins).
    
    **Critical:** Uses HLC instead of wall-clock time to avoid drift issues.
    **Merge:** Keep value with later timestamp
    """
    
    def __init__(self, replica_id: str, clock: HybridLogicalClock):
        self.replica_id = replica_id
        self.clock = clock
        self.value: Optional[V] = None
        self.timestamp: Optional[HLCTimestamp] = None
    
    def set(self, value: V) -> None:
        """Set register value with current HLC timestamp"""
        self.value = value
        self.timestamp = self.clock.now()
    
    def get(self) -> Optional[V]:
        """Get current register value"""
        return self.value
    
    def merge(self, other: 'LWWRegister[V]') -> 'LWWRegister[V]':
        """Merge with another LWW-Register"""
        result = LWWRegister[V](self.replica_id, self.clock)
        
        if self.timestamp is None and other.timestamp is None:
            # Both empty
            result.value = None
            result.timestamp = None
        elif self.timestamp is None:
            # Other has value
            result.value = other.value
            result.timestamp = other.timestamp
        elif other.timestamp is None:
            # We have value
            result.value = self.value
            result.timestamp = self.timestamp
        elif other.timestamp > self.timestamp:
            # Other is newer
            result.value = other.value
            result.timestamp = other.timestamp
        else:
            # We are newer (or equal - use replica_id as tiebreaker)
            if other.timestamp == self.timestamp:
                # Deterministic tiebreaker
                if other.replica_id > self.replica_id:
                    result.value = other.value
                    result.timestamp = other.timestamp
                else:
                    result.value = self.value
                    result.timestamp = self.timestamp
            else:
                result.value = self.value
                result.timestamp = self.timestamp
        
        # Update clock based on merged timestamp
        if result.timestamp:
            self.clock.update(result.timestamp)
        
        return result
    
    def __repr__(self) -> str:
        return f"LWWRegister(value={self.value}, ts={self.timestamp})"


class MVRegister(Generic[V]):
    """
    Multi-Value Register (MV-Register).
    
    A register that stores multiple concurrent values when conflicts occur.
    Application must resolve conflicts.
    
    **Merge:** Keep all values with non-dominated timestamps
    **Query:** Returns set of concurrent values
    """
    
    def __init__(self, replica_id: str, clock: HybridLogicalClock):
        self.replica_id = replica_id
        self.clock = clock
        # Map from value to timestamp
        self.values: Dict[V, HLCTimestamp] = {}
    
    def set(self, value: V) -> None:
        """Set register value (overwrites all previous values)"""
        timestamp = self.clock.now()
        self.values = {value: timestamp}
    
    def get(self) -> Set[V]:
        """Get current set of concurrent values"""
        return set(self.values.keys())
    
    def merge(self, other: 'MVRegister[V]') -> 'MVRegister[V]':
        """Merge with another MV-Register"""
        result = MVRegister[V](self.replica_id, self.clock)
        
        # Combine all values
        all_values = {}
        all_values.update(self.values)
        
        for value, timestamp in other.values.items():
            if value in all_values:
                # Keep the one with later timestamp
                if timestamp > all_values[value]:
                    all_values[value] = timestamp
            else:
                all_values[value] = timestamp
        
        # Remove dominated values (values with earlier timestamps)
        max_timestamp = max(all_values.values()) if all_values else None
        
        if max_timestamp:
            result.values = {
                v: ts for v, ts in all_values.items()
                if ts == max_timestamp
            }
        
        # Update clock
        for ts in result.values.values():
            self.clock.update(ts)
        
        return result
    
    def __repr__(self) -> str:
        return f"MVRegister(values={set(self.values.keys())})"


class ORSet(Generic[T]):
    """
    Observed-Remove Set (OR-Set).
    
    A set that allows both adds and removes, and elements can be re-added
    after removal. Uses unique tags (HLC timestamps) to track element versions.
    
    **Merge:** Union of elements, keeping only non-removed versions
    **Add bias:** If concurrent add and remove, add wins
    """
    
    def __init__(self, replica_id: str, clock: HybridLogicalClock):
        self.replica_id = replica_id
        self.clock = clock
        # Map from element to set of timestamps (versions)
        self.elements: Dict[T, Set[HLCTimestamp]] = defaultdict(set)
        self.removed: Dict[T, Set[HLCTimestamp]] = defaultdict(set)
    
    def add(self, element: T) -> None:
        """Add element with new unique timestamp"""
        timestamp = self.clock.now()
        self.elements[element].add(timestamp)
    
    def remove(self, element: T) -> None:
        """Remove all observed versions of element"""
        if element in self.elements:
            # Move all current versions to removed
            self.removed[element].update(self.elements[element])
            del self.elements[element]
    
    def contains(self, element: T) -> bool:
        """Check if element is in set (has non-removed versions)"""
        if element not in self.elements:
            return False
        
        # Element is in set if it has at least one non-removed version
        current_versions = self.elements[element]
        removed_versions = self.removed.get(element, set())
        
        return bool(current_versions - removed_versions)
    
    def merge(self, other: 'ORSet[T]') -> 'ORSet[T]':
        """Merge with another OR-Set"""
        result = ORSet[T](self.replica_id, self.clock)
        
        # Merge elements
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        
        for element in all_elements:
            # Union of versions
            versions = self.elements.get(element, set()) | other.elements.get(element, set())
            result.elements[element] = versions
        
        # Merge removed
        all_removed = set(self.removed.keys()) | set(other.removed.keys())
        
        for element in all_removed:
            removed = self.removed.get(element, set()) | other.removed.get(element, set())
            result.removed[element] = removed
        
        # Clean up removed versions
        for element in list(result.elements.keys()):
            active_versions = result.elements[element] - result.removed.get(element, set())
            if active_versions:
                result.elements[element] = active_versions
            else:
                del result.elements[element]
        
        return result
    
    def get_elements(self) -> Set[T]:
        """Get set of all elements with active versions"""
        return {e for e in self.elements if self.contains(e)}
    
    def __repr__(self) -> str:
        return f"ORSet({self.get_elements()})"


# Example usage
if __name__ == "__main__":
    print("=== CRDT Examples ===\n")
    
    # Example 1: G-Counter
    print("Example 1: G-Counter")
    c1 = GCounter("replica1")
    c2 = GCounter("replica2")
    
    c1.increment(5)
    c2.increment(3)
    
    merged = c1.merge(c2)
    print(f"Counter values: c1={c1.value()}, c2={c2.value()}, merged={merged.value()}")
    print()
    
    # Example 2: LWW-Register with HLC
    print("Example 2: LWW-Register with HLC")
    clock1 = HybridLogicalClock("node1")
    clock2 = HybridLogicalClock("node2")
    
    r1 = LWWRegister[str]("node1", clock1)
    r2 = LWWRegister[str]("node2", clock2)
    
    r1.set("value_from_node1")
    r2.set("value_from_node2")
    
    merged_r = r1.merge(r2)
    print(f"Register values: r1={r1.get()}, r2={r2.get()}, merged={merged_r.get()}")
    print(f"Winner timestamp: {merged_r.timestamp}")
    print()
    
    # Example 3: OR-Set
    print("Example 3: OR-Set")
    clock3 = HybridLogicalClock("node3")
    clock4 = HybridLogicalClock("node4")
    
    s1 = ORSet[str]("node3", clock3)
    s2 = ORSet[str]("node4", clock4)
    
    s1.add("apple")
    s1.add("banana")
    s2.add("cherry")
    s2.add("banana")  # Concurrent add
    
    s1.remove("banana")  # Node1 removes banana
    
    merged_s = s1.merge(s2)
    print(f"Set1: {s1.get_elements()}")
    print(f"Set2: {s2.get_elements()}")
    print(f"Merged (banana re-added): {merged_s.get_elements()}")
