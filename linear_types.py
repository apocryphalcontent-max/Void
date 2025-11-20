"""
Linear Types System for Void-State

Implements linear types to ensure resources are consumed exactly once,
preventing resource leaks in complex control flows.

Linear types enforce that a value must be used exactly once - it cannot be
duplicated or dropped. This is crucial for managing capabilities, file handles,
network connections, and other system resources.

References:
- "Linear Types Can Change the World!" (Wadler, 1990)
- Rust's ownership system
- Substructural type systems
"""

from typing import TypeVar, Generic, Callable, Any, Optional
from dataclasses import dataclass
import threading
import traceback
import sys

T = TypeVar('T')


class LinearTypeError(Exception):
    """Exception raised when linear type constraints are violated"""
    pass


class LinearResource(Generic[T]):
    """
    A linear resource that must be consumed exactly once.
    
    This wrapper ensures that wrapped resources cannot be used multiple times
    or dropped without consumption. The resource must be explicitly consumed
    via the consume() method.
    
    Example:
        >>> resource = LinearResource(database_connection)
        >>> conn = resource.consume()  # Must call this
        >>> # resource is now consumed and cannot be used again
    """
    
    def __init__(self, value: T, name: Optional[str] = None):
        """
        Create a linear resource.
        
        Args:
            value: The resource to wrap
            name: Optional name for debugging
        """
        self._value = value
        self._consumed = False
        self._name = name or f"LinearResource({type(value).__name__})"
        self._creation_stack = ''.join(traceback.format_stack()[:-1])
        self._lock = threading.Lock()
    
    def consume(self) -> T:
        """
        Consume the resource, returning its value.
        
        This can only be called once. Subsequent calls will raise an error.
        
        Returns:
            The wrapped value
            
        Raises:
            LinearTypeError: If already consumed
        """
        with self._lock:
            if self._consumed:
                raise LinearTypeError(
                    f"LINEAR TYPE VIOLATION: {self._name} already consumed. "
                    f"Resource cannot be used multiple times."
                )
            self._consumed = True
            return self._value
    
    def is_consumed(self) -> bool:
        """Check if resource has been consumed"""
        with self._lock:
            return self._consumed
    
    def __del__(self):
        """
        Destructor that enforces consumption.
        
        This is called when the object is garbage collected. If the resource
        hasn't been consumed, it indicates a resource leak.
        """
        if not self._consumed:
            error_msg = (
                f"\n{'='*70}\n"
                f"LINEAR RESOURCE LEAK DETECTED!\n"
                f"{'='*70}\n"
                f"Resource: {self._name}\n"
                f"Created at:\n{self._creation_stack}\n"
                f"{'='*70}\n"
                f"This resource was dropped without consumption.\n"
                f"Linear resources must be explicitly consumed via .consume()\n"
                f"{'='*70}\n"
            )
            
            # In development, crash hard. In production, log the error.
            if __debug__:  # Debug mode (python without -O flag)
                # Use sys.exit to ensure the error is visible
                sys.stderr.write(error_msg)
                sys.stderr.flush()
                # Raise exception that will be logged but won't crash in __del__
                # Note: Exceptions in __del__ are printed but don't propagate
                raise LinearTypeError(f"LINEAR RESOURCE LEAK: {self._name} dropped without consumption")
            else:
                # Production: Just log the error
                sys.stderr.write(f"WARNING: {error_msg}")
                sys.stderr.flush()
    
    def __repr__(self) -> str:
        status = "consumed" if self._consumed else "unconsumed"
        return f"LinearResource({self._name}, {status})"


@dataclass
class LinearFunction(Generic[T]):
    """
    A function that consumes linear resources.
    
    This wrapper ensures that functions properly consume their linear arguments.
    """
    func: Callable[[T], Any]
    
    def __call__(self, resource: LinearResource[T]) -> Any:
        """
        Call the wrapped function with a linear resource.
        
        Args:
            resource: Linear resource to consume
            
        Returns:
            Result of the function
        """
        value = resource.consume()
        return self.func(value)


class LinearContext:
    """
    Context manager for tracking linear resources.
    
    This can be used to ensure all linear resources created within a context
    are properly consumed before exiting.
    
    Example:
        >>> with LinearContext() as ctx:
        ...     resource = ctx.create(database_connection)
        ...     result = resource.consume()
        ...     # Context ensures all resources are consumed
    """
    
    def __init__(self):
        self.resources: list[LinearResource] = []
        self._lock = threading.Lock()
    
    def create(self, value: T, name: Optional[str] = None) -> LinearResource[T]:
        """
        Create and track a linear resource.
        
        Args:
            value: The value to wrap
            name: Optional name for debugging
            
        Returns:
            A tracked linear resource
        """
        resource = LinearResource(value, name)
        with self._lock:
            self.resources.append(resource)
        return resource
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Check that all resources were consumed.
        
        Raises:
            LinearTypeError: If any resources remain unconsumed
        """
        with self._lock:
            unconsumed = [r for r in self.resources if not r.is_consumed()]
            if unconsumed:
                names = [r._name for r in unconsumed]
                raise LinearTypeError(
                    f"LINEAR TYPE VIOLATION: {len(unconsumed)} resource(s) "
                    f"not consumed in context: {names}"
                )
        return False  # Don't suppress exceptions


# ============================================================================
# AFFINE TYPES (USE-AT-MOST-ONCE)
# ============================================================================

class AffineResource(Generic[T]):
    """
    An affine resource that can be used at most once.
    
    Unlike linear resources (must use exactly once), affine resources
    can be dropped without consumption (use-at-most-once). However,
    they still cannot be duplicated.
    
    This is useful for optional resources or cleanup handlers.
    """
    
    def __init__(self, value: T, name: Optional[str] = None):
        self._value = value
        self._consumed = False
        self._name = name or f"AffineResource({type(value).__name__})"
        self._lock = threading.Lock()
    
    def consume(self) -> T:
        """Consume the resource (can only be called once)"""
        with self._lock:
            if self._consumed:
                raise LinearTypeError(
                    f"AFFINE TYPE VIOLATION: {self._name} already consumed"
                )
            self._consumed = True
            return self._value
    
    def is_consumed(self) -> bool:
        """Check if resource has been consumed"""
        with self._lock:
            return self._consumed
    
    def __repr__(self) -> str:
        status = "consumed" if self._consumed else "unconsumed"
        return f"AffineResource({self._name}, {status})"


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def _example_usage():
    """Demonstrate linear types usage"""
    
    print("Linear Types Example\n" + "="*50)
    
    # Example 1: Correct usage
    print("\n1. Correct usage:")
    resource = LinearResource("database_connection", name="db_conn")
    print(f"   Created: {resource}")
    conn = resource.consume()
    print(f"   Consumed: {conn}")
    print(f"   Status: {resource}")
    
    # Example 2: Using context manager
    print("\n2. Using LinearContext:")
    with LinearContext() as ctx:
        r1 = ctx.create("resource1")
        r2 = ctx.create("resource2")
        print(f"   Created resources: {r1}, {r2}")
        v1 = r1.consume()
        v2 = r2.consume()
        print(f"   All consumed: {v1}, {v2}")
    print("   Context exited successfully")
    
    # Example 3: Affine resources (can be dropped)
    print("\n3. Affine resources (use-at-most-once):")
    affine = AffineResource("optional_resource")
    print(f"   Created: {affine}")
    # Can be dropped without consuming (unlike linear resources)
    del affine
    print("   Dropped without consuming (OK for affine types)")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    _example_usage()
