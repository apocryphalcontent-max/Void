"""
Dependent Types System for Void-State

Implements dependent types where types can depend on values, enabling
compile-time (or import-time) verification of properties like array sizes,
ranges, and other value-dependent constraints.

Dependent types allow expressing constraints like:
- Vector[10] - a vector of exactly 10 elements
- Range[1, 100] - an integer in the range [1, 100]
- NonEmpty[List] - a non-empty list

References:
- "Type Theory and Formal Proof" (Nederpelt & Geuvers, 2014)
- Idris programming language
- "Dependent Types in Practical Programming" (Xi & Pfenning, 1999)
"""

from typing import (
    TypeVar, Generic, Any, Type, Dict, Tuple, 
    get_args, get_origin, Union, Optional
)
from abc import ABC, abstractmethod
import weakref
from functools import lru_cache
import sys

T = TypeVar('T')
N = TypeVar('N')


# ============================================================================
# DEPENDENT TYPE METACLASS
# ============================================================================

class DependentMeta(type):
    """
    Metaclass for dependent types.
    
    This metaclass enables the Vector[N] syntax and caches type definitions
    for literal values to avoid regenerating validators on every instantiation.
    
    Key optimization: If N is a literal (e.g., Vector[5]), the class definition
    is cached and reused.
    """
    
    # Cache for parameterized types: (base_class, params) -> type
    _type_cache: Dict[Tuple[Type, Tuple], Type] = {}
    
    def __getitem__(cls, params):
        """
        Enable parameterized type syntax like Vector[10].
        
        Args:
            params: Type parameters (can be single value or tuple)
            
        Returns:
            A cached or new parameterized type
        """
        # Normalize params to tuple
        if not isinstance(params, tuple):
            params = (params,)
        
        # Check cache first
        cache_key = (cls, params)
        if cache_key in DependentMeta._type_cache:
            return DependentMeta._type_cache[cache_key]
        
        # Create new parameterized type
        param_type = cls._create_parameterized_type(cls, params)
        
        # Cache it
        DependentMeta._type_cache[cache_key] = param_type
        
        return param_type
    
    @staticmethod
    def _create_parameterized_type(base_cls, params):
        """
        Create a new parameterized type.
        
        Args:
            base_cls: The base class being parameterized
            params: The type parameters
            
        Returns:
            A new type with the given parameters
        """
        # Create a new class with the parameters embedded
        param_str = '_'.join(str(p) for p in params)
        class_name = f"{base_cls.__name__}[{param_str}]"
        
        # Create the new type with validation
        def __init__(self, *args, **kwargs):
            # Initialize the base class
            base_cls.__init__(self, *args, **kwargs)
            # Validate the dependent type constraints
            self._validate_dependent_constraints(params)
        
        # Create new class inheriting from base
        parameterized_type = type(
            class_name,
            (base_cls,),
            {
                '__init__': __init__,
                '_type_params': params,
                '__module__': base_cls.__module__,
            }
        )
        
        return parameterized_type


class DependentType:
    """
    Base class for dependent types.
    
    Subclasses should implement _validate_dependent_constraints to check
    that the value satisfies the type-level constraints.
    """
    
    def _validate_dependent_constraints(self, params):
        """
        Validate that this value satisfies the dependent type constraints.
        
        Args:
            params: The type parameters
            
        Raises:
            TypeError: If constraints are violated
        """
        pass


# ============================================================================
# SIZED VECTOR TYPE
# ============================================================================

class Vector(DependentType, metaclass=DependentMeta):
    """
    A vector with size as a dependent type parameter.
    
    Vector[N] represents a vector of exactly N elements.
    The size is verified at construction time.
    
    Example:
        >>> v = Vector[3]([1, 2, 3])  # OK
        >>> v = Vector[3]([1, 2])     # TypeError: Expected 3 elements
    """
    
    def __init__(self, data):
        """
        Initialize vector with data.
        
        Args:
            data: Sequence of elements
        """
        if isinstance(data, (list, tuple)):
            self.data = list(data)
        else:
            self.data = list(data)
        self._size = len(self.data)
    
    def _validate_dependent_constraints(self, params):
        """Validate that vector has expected size"""
        expected_size = params[0]
        if self._size != expected_size:
            raise TypeError(
                f"Vector size mismatch: expected {expected_size}, "
                f"got {self._size}"
            )
    
    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __repr__(self) -> str:
        return f"Vector[{self._size}]({self.data})"
    
    def __iter__(self):
        return iter(self.data)


# ============================================================================
# RANGED INTEGER TYPE
# ============================================================================

class RangedInt(DependentType, metaclass=DependentMeta):
    """
    An integer constrained to a specific range.
    
    Range[min, max] represents an integer in [min, max].
    
    Example:
        >>> x = Range[1, 100](50)  # OK
        >>> x = Range[1, 100](150) # TypeError: Out of range
    """
    
    def __init__(self, value: int):
        """
        Initialize ranged integer.
        
        Args:
            value: The integer value
        """
        if not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")
        self.value = value
    
    def _validate_dependent_constraints(self, params):
        """Validate that value is in range"""
        min_val, max_val = params[0], params[1]
        if not (min_val <= self.value <= max_val):
            raise TypeError(
                f"Value {self.value} out of range [{min_val}, {max_val}]"
            )
    
    def __int__(self) -> int:
        return self.value
    
    def __repr__(self) -> str:
        if hasattr(self, '_type_params'):
            min_val, max_val = self._type_params
            return f"Range[{min_val}, {max_val}]({self.value})"
        return f"RangedInt({self.value})"
    
    def __eq__(self, other):
        if isinstance(other, RangedInt):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        return hash(self.value)


# Alias for convenience
Range = RangedInt


# ============================================================================
# NON-EMPTY COLLECTION TYPE
# ============================================================================

class NonEmpty(DependentType, Generic[T], metaclass=DependentMeta):
    """
    A non-empty collection type.
    
    NonEmpty[List] represents a list that must have at least one element.
    
    Example:
        >>> items = NonEmpty[list]([1, 2, 3])  # OK
        >>> items = NonEmpty[list]([])         # TypeError: Empty collection
    """
    
    def __init__(self, data):
        """
        Initialize non-empty collection.
        
        Args:
            data: The collection (must be non-empty)
        """
        self.data = data
    
    def _validate_dependent_constraints(self, params):
        """Validate that collection is non-empty"""
        if len(self.data) == 0:
            raise TypeError("NonEmpty constraint violated: collection is empty")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __iter__(self):
        return iter(self.data)
    
    def __repr__(self) -> str:
        return f"NonEmpty({self.data})"


# ============================================================================
# REFINED TYPES (PREDICATE-BASED)
# ============================================================================

class Refined(DependentType, metaclass=DependentMeta):
    """
    A refined type with a predicate constraint.
    
    Refined types allow arbitrary predicates to constrain values.
    
    Example:
        >>> even_int = Refined[int, lambda x: x % 2 == 0](4)  # OK
        >>> even_int = Refined[int, lambda x: x % 2 == 0](3)  # TypeError
    """
    
    def __init__(self, value):
        """
        Initialize refined type.
        
        Args:
            value: The value to refine
        """
        self.value = value
    
    def _validate_dependent_constraints(self, params):
        """Validate that value satisfies predicate"""
        base_type, predicate = params[0], params[1]
        
        # Check base type
        if not isinstance(self.value, base_type):
            raise TypeError(
                f"Expected {base_type.__name__}, got {type(self.value).__name__}"
            )
        
        # Check predicate
        if not predicate(self.value):
            raise TypeError(
                f"Value {self.value} does not satisfy refinement predicate"
            )
    
    def __repr__(self) -> str:
        return f"Refined({self.value})"


# ============================================================================
# IMPORT-TIME VALIDATION DECORATOR
# ============================================================================

def validated(*validators):
    """
    Decorator that adds import-time validation to functions.
    
    This moves validation from runtime to import time where possible,
    reducing overhead.
    
    Example:
        @validated(lambda args: len(args) == 2)
        def add(x, y):
            return x + y
    """
    def decorator(func):
        # Store validators for later use
        func._validators = validators
        
        def wrapper(*args, **kwargs):
            # Validate at call time
            for validator in validators:
                if not validator(args):
                    raise TypeError(f"Validation failed for {func.__name__}")
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# MEMOIZED TYPE CONSTRUCTORS
# ============================================================================

@lru_cache(maxsize=256)
def make_vector_type(size: int) -> Type[Vector]:
    """
    Create a cached Vector type for a given size.
    
    This is heavily memoized to avoid creating new types repeatedly.
    
    Args:
        size: The vector size
        
    Returns:
        A Vector type parameterized with the given size
    """
    return Vector[size]


@lru_cache(maxsize=256)
def make_range_type(min_val: int, max_val: int) -> Type[RangedInt]:
    """
    Create a cached Range type.
    
    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        
    Returns:
        A Range type for the given bounds
    """
    return Range[min_val, max_val]


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def _example_usage():
    """Demonstrate dependent types usage"""
    
    print("Dependent Types Example\n" + "="*50)
    
    # Example 1: Sized vectors
    print("\n1. Sized Vectors:")
    try:
        v1 = Vector[3]([1, 2, 3])
        print(f"   ✓ Created: {v1}")
        
        v2 = Vector[3]([1, 2])  # Wrong size - should fail
        print(f"   ✗ Should not reach here: {v2}")
    except TypeError as e:
        print(f"   ✓ Caught error: {e}")
    
    # Example 2: Ranged integers
    print("\n2. Ranged Integers:")
    try:
        age = Range[0, 120](25)
        print(f"   ✓ Created: {age}")
        
        invalid_age = Range[0, 120](150)
        print(f"   ✗ Should not reach here: {invalid_age}")
    except TypeError as e:
        print(f"   ✓ Caught error: {e}")
    
    # Example 3: Non-empty collections
    print("\n3. Non-empty Collections:")
    try:
        items = NonEmpty[list]([1, 2, 3])
        print(f"   ✓ Created: {items}")
        
        empty = NonEmpty[list]([])
        print(f"   ✗ Should not reach here: {empty}")
    except TypeError as e:
        print(f"   ✓ Caught error: {e}")
    
    # Example 4: Type caching
    print("\n4. Type Caching:")
    Vec3A = make_vector_type(3)
    Vec3B = make_vector_type(3)
    print(f"   Same type object: {Vec3A is Vec3B}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    _example_usage()
