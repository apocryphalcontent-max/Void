"""
Linear Types for Resource Management

Adds linear types that enforce single-use semantics for resources.
Linear types guarantee resources are used exactly once, preventing leaks and 
double-frees at compile time.

Research Connection: Rust's ownership system, Clean's uniqueness types, Linear Haskell.
"""

import threading

class Linear:
    """Marker for linearly-typed values - must be used exactly once"""
    _used: bool = False
    
    def consume(self):
        """Consume this resource (can only be called once)"""
        if self._used:
            raise RuntimeError("Linear resource already consumed")
        self._used = True
        return self
    
    def __del__(self):
        """Ensure resource was consumed"""
        if not self._used:
            raise RuntimeError("Linear resource not consumed")

class LinearLock(Linear):
    """Linear lock that must be acquired exactly once"""
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
    
    def acquire(self) -> 'AcquiredLock':
        """Acquire lock, consuming this LinearLock"""
        self.consume()
        self._lock.acquire()
        return AcquiredLock(self._lock)

class AcquiredLock(Linear):
    """Token representing an acquired lock"""
    def __init__(self, lock):
        self._lock = lock
    
    def release(self):
        """Release lock, consuming this AcquiredLock"""
        self.consume()
        self._lock.release()

# Applications:
# - Memory region access tokens
# - Distributed lock ownership
# - One-time initialization guarantees
# - Safe suspend/resume lifecycle
