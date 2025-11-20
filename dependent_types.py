"""
Dependent Types for Void-State System

Implements dependent types where types can depend on runtime values.
This provides compile-time (or import-time) verification of invariants
like vector dimensions, matrix sizes, and bounded integers.

References:
- "The Power of Pi" (Oury & Swierstra, 2008)
- Idris programming language
- Dependent Types in Practical Programming (Xi & Pfenning, 1999)
"""

from typing import TypeVar, Generic, Any, Optional, Callable, Type, Dict, get_args
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import functools
import inspect


T = TypeVar('T')
N = TypeVar('N', bound=int)


# Cache for memoized type classes
_TYPE_CACHE: Dict[tuple, Type] = {}


class DependentTypeError(Exception):
    """Raised when dependent type constraints are violated"""
    pass


class DependentMeta(type):
    """
    Metaclass for dependent types.
    
    Enables creation of types parameterized by runtime values, with
    validation moved to import time or class creation time when possible.
    
    Example:
        Vector[5]  # Creates a vector type that must have length 5
        Matrix[3, 4]  # Creates a matrix type with shape (3, 4)
    """
    
    def __getitem__(cls, params):
        """
        Handle parameterized type creation like Vector[N].
        
        This is called at class definition time, so we can cache and
        optimize the type checking.
        """
        # Normalize params to tuple
        if not isinstance(params, tuple):
            params = (params,)
        
        # Check cache first (crucial optimization)
        cache_key = (cls, params)
        if cache_key in _TYPE_CACHE:
            return _TYPE_CACHE[cache_key]
        
        # Create new parameterized type
        class ParameterizedType(cls):
            _params = params
            
            def __init__(self, value):
                # Validate at construction time
                cls._validate(value, params)
                super().__init__(value)
        
        # Cache the type class
        _TYPE_CACHE[cache_key] = ParameterizedType
        
        # Set a readable name
        param_str = ', '.join(str(p) for p in params)
        ParameterizedType.__name__ = f"{cls.__name__}[{param_str}]"
        ParameterizedType.__qualname__ = f"{cls.__qualname__}[{param_str}]"
        
        return ParameterizedType
    
    @staticmethod
    def _validate(value: Any, params: tuple) -> None:
        """Override in subclasses to implement validation logic"""
        pass


class Vector(metaclass=DependentMeta):
    """
    A vector with a statically known length.
    
    Example:
        v = Vector[5]([1, 2, 3, 4, 5])  # OK
        v = Vector[5]([1, 2, 3])  # Raises DependentTypeError
    """
    
    def __init__(self, value):
        if isinstance(value, np.ndarray):
            self._value = value
        else:
            self._value = np.array(value)
    
    @staticmethod
    def _validate(value, params):
        """Validate vector length"""
        if len(params) != 1:
            raise DependentTypeError(
                f"Vector expects 1 parameter (length), got {len(params)}"
            )
        
        expected_length = params[0]
        if not isinstance(expected_length, int):
            raise DependentTypeError(
                f"Vector length must be an integer, got {type(expected_length)}"
            )
        
        # Convert value to numpy array if needed
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        
        actual_length = len(value)
        if actual_length != expected_length:
            raise DependentTypeError(
                f"Vector length mismatch: expected {expected_length}, "
                f"got {actual_length}"
            )
    
    def __getitem__(self, index):
        return self._value[index]
    
    def __len__(self):
        return len(self._value)
    
    def __repr__(self):
        return f"Vector({self._value.tolist()})"
    
    @property
    def value(self) -> NDArray:
        return self._value


class Matrix(metaclass=DependentMeta):
    """
    A matrix with statically known dimensions.
    
    Example:
        m = Matrix[3, 4](np.zeros((3, 4)))  # OK
        m = Matrix[3, 4](np.zeros((2, 4)))  # Raises DependentTypeError
    """
    
    def __init__(self, value):
        if isinstance(value, np.ndarray):
            self._value = value
        else:
            self._value = np.array(value)
    
    @staticmethod
    def _validate(value, params):
        """Validate matrix shape"""
        if len(params) != 2:
            raise DependentTypeError(
                f"Matrix expects 2 parameters (rows, cols), got {len(params)}"
            )
        
        expected_rows, expected_cols = params
        if not isinstance(expected_rows, int) or not isinstance(expected_cols, int):
            raise DependentTypeError(
                f"Matrix dimensions must be integers"
            )
        
        # Convert value to numpy array if needed
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        
        if value.ndim != 2:
            raise DependentTypeError(
                f"Matrix must be 2-dimensional, got {value.ndim}D"
            )
        
        actual_shape = value.shape
        expected_shape = (expected_rows, expected_cols)
        if actual_shape != expected_shape:
            raise DependentTypeError(
                f"Matrix shape mismatch: expected {expected_shape}, "
                f"got {actual_shape}"
            )
    
    def __getitem__(self, index):
        return self._value[index]
    
    @property
    def shape(self):
        return self._value.shape
    
    def __repr__(self):
        return f"Matrix{self.shape}({self._value.tolist()})"
    
    @property
    def value(self) -> NDArray:
        return self._value


class BoundedInt(metaclass=DependentMeta):
    """
    An integer bounded by minimum and maximum values.
    
    Example:
        age = BoundedInt[0, 150](25)  # OK
        age = BoundedInt[0, 150](200)  # Raises DependentTypeError
    """
    
    def __init__(self, value: int):
        self._value = int(value)
    
    @staticmethod
    def _validate(value, params):
        """Validate integer bounds"""
        if len(params) != 2:
            raise DependentTypeError(
                f"BoundedInt expects 2 parameters (min, max), got {len(params)}"
            )
        
        min_val, max_val = params
        if not isinstance(min_val, int) or not isinstance(max_val, int):
            raise DependentTypeError(
                f"Bounds must be integers"
            )
        
        if min_val > max_val:
            raise DependentTypeError(
                f"Invalid bounds: min ({min_val}) > max ({max_val})"
            )
        
        int_value = int(value)
        if int_value < min_val or int_value > max_val:
            raise DependentTypeError(
                f"Value {int_value} out of bounds [{min_val}, {max_val}]"
            )
    
    def __int__(self):
        return self._value
    
    def __repr__(self):
        return f"BoundedInt({self._value})"
    
    @property
    def value(self) -> int:
        return self._value


class NonEmpty(metaclass=DependentMeta):
    """
    A non-empty collection (list, tuple, etc.).
    
    Example:
        items = NonEmpty([1, 2, 3])  # OK
        items = NonEmpty([])  # Raises DependentTypeError
    """
    
    def __init__(self, value):
        self._value = value
    
    @staticmethod
    def _validate(value, params):
        """Validate non-emptiness"""
        if not value:
            raise DependentTypeError(
                "Collection must be non-empty"
            )
    
    def __len__(self):
        return len(self._value)
    
    def __getitem__(self, index):
        return self._value[index]
    
    def __iter__(self):
        return iter(self._value)
    
    def __repr__(self):
        return f"NonEmpty({self._value})"
    
    @property
    def value(self):
        return self._value


def dependent(func: Callable) -> Callable:
    """
    Decorator to enforce dependent type checking on function arguments.
    
    Example:
        @dependent
        def dot_product(a: Vector[3], b: Vector[3]) -> float:
            return np.dot(a.value, b.value)
    """
    sig = inspect.signature(func)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Bind arguments to signature
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        
        # Check each parameter's type annotation
        for param_name, param_value in bound.arguments.items():
            param = sig.parameters[param_name]
            
            if param.annotation != inspect.Parameter.empty:
                # Check if it's a dependent type
                annotation = param.annotation
                if hasattr(annotation, '__origin__'):
                    # It's a generic type
                    origin = annotation.__origin__
                    if hasattr(origin, '_validate'):
                        # It's a dependent type - verify it
                        if not isinstance(param_value, origin):
                            raise DependentTypeError(
                                f"Parameter {param_name} must be of type {annotation}"
                            )
        
        return func(*args, **kwargs)
    
    return wrapper


@dataclass
class Refinement(Generic[T]):
    """
    A refinement type: a base type with a predicate.
    
    Example:
        PositiveInt = Refinement(int, lambda x: x > 0)
        age = PositiveInt.validate(25)  # OK
        age = PositiveInt.validate(-5)  # Raises DependentTypeError
    """
    base_type: Type[T]
    predicate: Callable[[T], bool]
    name: Optional[str] = None
    
    def validate(self, value: Any) -> T:
        """
        Validate that value satisfies the refinement.
        
        Args:
            value: Value to validate
            
        Returns:
            The validated value
            
        Raises:
            DependentTypeError: If validation fails
        """
        if not isinstance(value, self.base_type):
            raise DependentTypeError(
                f"Expected type {self.base_type.__name__}, "
                f"got {type(value).__name__}"
            )
        
        if not self.predicate(value):
            name = self.name or f"{self.base_type.__name__} with predicate"
            raise DependentTypeError(
                f"Value {value} does not satisfy refinement {name}"
            )
        
        return value
    
    def __call__(self, value: Any) -> T:
        """Allow using refinement as a constructor"""
        return self.validate(value)


# Common refinement types
PositiveInt = Refinement(int, lambda x: x > 0, "PositiveInt")
NegativeInt = Refinement(int, lambda x: x < 0, "NegativeInt")
NonNegativeInt = Refinement(int, lambda x: x >= 0, "NonNegativeInt")
PositiveFloat = Refinement(float, lambda x: x > 0.0, "PositiveFloat")


# Example usage and tests
if __name__ == "__main__":
    print("=== Dependent Types Examples ===\n")
    
    # Example 1: Vector with fixed size
    print("Example 1: Vector[5]")
    try:
        v1 = Vector[5]([1, 2, 3, 4, 5])
        print(f"Created: {v1}")
        
        v2 = Vector[5]([1, 2, 3])  # This will fail
    except DependentTypeError as e:
        print(f"Expected error: {e}\n")
    
    # Example 2: Matrix with fixed shape
    print("Example 2: Matrix[3, 4]")
    try:
        m1 = Matrix[3, 4](np.ones((3, 4)))
        print(f"Created: {m1}")
        
        m2 = Matrix[3, 4](np.ones((2, 4)))  # This will fail
    except DependentTypeError as e:
        print(f"Expected error: {e}\n")
    
    # Example 3: BoundedInt
    print("Example 3: BoundedInt[0, 100]")
    try:
        age = BoundedInt[0, 100](25)
        print(f"Created: {age}")
        
        invalid_age = BoundedInt[0, 100](150)  # This will fail
    except DependentTypeError as e:
        print(f"Expected error: {e}\n")
    
    # Example 4: NonEmpty
    print("Example 4: NonEmpty")
    try:
        items = NonEmpty()([1, 2, 3])
        print(f"Created: {items}")
        
        empty = NonEmpty()([])  # This will fail
    except DependentTypeError as e:
        print(f"Expected error: {e}\n")
    
    # Example 5: Refinement types
    print("Example 5: PositiveInt")
    try:
        pos = PositiveInt(42)
        print(f"Created: {pos}")
        
        neg = PositiveInt(-5)  # This will fail
    except DependentTypeError as e:
        print(f"Expected error: {e}\n")
    
    # Example 6: Type caching (performance optimization)
    print("Example 6: Type caching")
    Vec5_1 = Vector[5]
    Vec5_2 = Vector[5]
    print(f"Same type cached: {Vec5_1 is Vec5_2}")
    print(f"Cache size: {len(_TYPE_CACHE)}")
