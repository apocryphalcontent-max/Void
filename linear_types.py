"""
Linear Types for Void-State System

Implements linear types to prevent resource leaks by ensuring resources
are consumed exactly once. This is critical for capability-based security
and memory management in the Void system.

References:
- Linear Types Can Change the World (Wadler, 1990)
- A Taste of Linear Logic (Wadler, 1993)
- Rust's affine type system
"""

from typing import TypeVar, Generic, Optional, Callable, Any
from dataclasses import dataclass
import threading
import warnings


T = TypeVar('T')


class LinearTypeError(Exception):
    """Raised when linear type invariants are violated"""
    pass


class LinearResource(Generic[T]):
    """
    A linear resource that must be consumed exactly once.
    
    This wrapper ensures that resources are not duplicated, leaked, or
    used after consumption. Critical for capabilities and memory safety.
    
    Example:
        resource = LinearResource(file_handle)
        data = resource.consume()  # OK
        # resource.consume()  # Would raise LinearTypeError
    """
    
    def __init__(self, value: T):
        """
        Create a linear resource.
        
        Args:
            value: The underlying resource to protect
        """
        self._value = value
        self._consumed = False
        self._lock = threading.Lock()
    
    def consume(self) -> T:
        """
        Consume the resource, extracting its value.
        
        This can only be called once. Subsequent calls will raise LinearTypeError.
        
        Returns:
            The underlying resource value
            
        Raises:
            LinearTypeError: If resource was already consumed
        """
        with self._lock:
            if self._consumed:
                raise LinearTypeError(
                    "LINEAR TYPE VIOLATION: Resource has already been consumed. "
                    "Linear resources must be used exactly once."
                )
            self._consumed = True
            return self._value
    
    def is_consumed(self) -> bool:
        """Check if resource has been consumed"""
        return self._consumed
    
    def __del__(self):
        """
        Destructor that enforces linear type invariant.
        
        If a linear resource is garbage collected without being consumed,
        this indicates a resource leak. In development, this should crash;
        in production, it should log a critical error.
        """
        if not self._consumed:
            # This is a critical error - a linear resource was leaked
            # In development: crash hard
            # In production: log and continue (but this is still a bug)
            error_msg = (
                "LINEAR RESOURCE LEAK: Resource dropped without consumption. "
                "This indicates a bug in the code - linear resources must be "
                "explicitly consumed before going out of scope."
            )
            
            # Check if we're in development mode (can be configured)
            import os
            if os.environ.get('VOID_ENV') == 'development':
                raise MemoryError(error_msg)
            else:
                # In production, warn but don't crash
                warnings.warn(error_msg, ResourceWarning, stacklevel=2)
    
    def __repr__(self) -> str:
        status = "consumed" if self._consumed else "unconsumed"
        return f"LinearResource({status})"


class LinearBox(Generic[T]):
    """
    An affine type that can be moved but not copied.
    
    Similar to LinearResource but allows inspection before consumption.
    Once moved, the original box becomes invalid.
    """
    
    def __init__(self, value: T):
        self._value: Optional[T] = value
        self._moved = False
        self._lock = threading.Lock()
    
    def take(self) -> T:
        """
        Move the value out of the box, invalidating it.
        
        Returns:
            The underlying value
            
        Raises:
            LinearTypeError: If value was already moved
        """
        with self._lock:
            if self._moved:
                raise LinearTypeError(
                    "AFFINE TYPE VIOLATION: Value has already been moved. "
                    "Cannot move from the same box twice."
                )
            if self._value is None:
                raise LinearTypeError("Box is empty")
            
            self._moved = True
            value = self._value
            self._value = None
            return value
    
    def peek(self) -> T:
        """
        Inspect the value without consuming it (immutable borrow).
        
        Returns:
            The underlying value
            
        Raises:
            LinearTypeError: If value was already moved
        """
        with self._lock:
            if self._moved:
                raise LinearTypeError(
                    "Cannot peek at moved value"
                )
            if self._value is None:
                raise LinearTypeError("Box is empty")
            return self._value
    
    def is_valid(self) -> bool:
        """Check if the box still contains a value"""
        return not self._moved and self._value is not None


@dataclass
class LinearContext:
    """
    Context manager for tracking linear resource usage.
    
    Ensures all linear resources within a context are properly consumed.
    """
    resources: list = None
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = []
    
    def track(self, resource: LinearResource[T]) -> LinearResource[T]:
        """Track a linear resource in this context"""
        self.resources.append(resource)
        return resource
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Verify all resources were consumed"""
        leaks = [r for r in self.resources if not r.is_consumed()]
        if leaks:
            raise LinearTypeError(
                f"LINEAR RESOURCE LEAK: {len(leaks)} resource(s) not consumed "
                f"before exiting context"
            )
        return False


def linear(func: Callable) -> Callable:
    """
    Decorator to enforce linear type checking on function parameters.
    
    Example:
        @linear
        def use_capability(cap: LinearResource[Capability]) -> Result:
            token = cap.consume()
            return perform_action(token)
    """
    def wrapper(*args, **kwargs):
        # Check for LinearResource arguments
        for arg in args:
            if isinstance(arg, LinearResource) and arg.is_consumed():
                raise LinearTypeError(
                    f"Cannot pass consumed LinearResource to {func.__name__}"
                )
        for arg in kwargs.values():
            if isinstance(arg, LinearResource) and arg.is_consumed():
                raise LinearTypeError(
                    f"Cannot pass consumed LinearResource to {func.__name__}"
                )
        
        return func(*args, **kwargs)
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# Example usage and tests
if __name__ == "__main__":
    # Example 1: Basic linear resource usage
    print("Example 1: Basic usage")
    resource = LinearResource("secret_token")
    token = resource.consume()
    print(f"Consumed: {token}")
    
    try:
        resource.consume()  # This will fail
    except LinearTypeError as e:
        print(f"Expected error: {e}")
    
    # Example 2: Context manager
    print("\nExample 2: Context manager")
    with LinearContext() as ctx:
        res1 = ctx.track(LinearResource(42))
        res2 = ctx.track(LinearResource("data"))
        
        val1 = res1.consume()
        val2 = res2.consume()
        print(f"Consumed in context: {val1}, {val2}")
    
    # Example 3: LinearBox
    print("\nExample 3: LinearBox")
    box = LinearBox([1, 2, 3])
    print(f"Peek: {box.peek()}")  # OK - immutable borrow
    data = box.take()  # Move out
    print(f"Taken: {data}")
    
    try:
        box.take()  # This will fail
    except LinearTypeError as e:
        print(f"Expected error: {e}")
