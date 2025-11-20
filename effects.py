"""
Algebraic Effects System for Void-State

Implements algebraic effects and handlers for dependency injection,
testing, and decoupling of side effects from business logic.

Algebraic effects provide:
1. Explicit effect declarations
2. Composable effect handlers
3. Modular reasoning about side effects
4. Deterministic testing via effect mocking

This enables:
- Time control for testing race conditions
- Network mocking for testing
- State management without globals
- Logging/monitoring decoupled from logic

References:
- "An Introduction to Algebraic Effects and Handlers" (Pretnar, 2015)
- "Programming with Algebraic Effects and Handlers" (Bauer & Pretnar, 2015)
- OCaml's effect system, Koka language
"""

from typing import TypeVar, Generic, Callable, Any, Optional, Dict, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum, auto
import time
from contextlib import contextmanager
import threading

T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# EFFECT DEFINITIONS
# ============================================================================

class Effect(ABC):
    """
    Base class for effects.
    
    An effect represents a request for a capability (time, state, I/O, etc.)
    that will be handled by an effect handler.
    """
    
    @abstractmethod
    def effect_name(self) -> str:
        """Return the name of this effect"""
        pass


@dataclass
class EffectResult(Generic[T]):
    """Result of performing an effect"""
    value: T
    continuation: Optional[Callable] = None


# ============================================================================
# COMMON EFFECTS
# ============================================================================

@dataclass
class AskCurrentTime(Effect):
    """Effect: Request current time"""
    
    def effect_name(self) -> str:
        return "AskCurrentTime"


@dataclass
class Sleep(Effect):
    """Effect: Sleep for duration"""
    duration: float
    
    def effect_name(self) -> str:
        return "Sleep"


@dataclass
class Log(Effect):
    """Effect: Log a message"""
    level: str
    message: str
    
    def effect_name(self) -> str:
        return "Log"


@dataclass
class GetState(Effect):
    """Effect: Get current state"""
    key: str
    
    def effect_name(self) -> str:
        return "GetState"


@dataclass
class SetState(Effect):
    """Effect: Set state"""
    key: str
    value: Any
    
    def effect_name(self) -> str:
        return "SetState"


@dataclass
class NetworkRequest(Effect):
    """Effect: Perform network request"""
    url: str
    method: str = "GET"
    data: Optional[Any] = None
    
    def effect_name(self) -> str:
        return "NetworkRequest"


@dataclass
class RandomValue(Effect):
    """Effect: Get random value in range"""
    min_val: float = 0.0
    max_val: float = 1.0
    
    def effect_name(self) -> str:
        return "RandomValue"


# ============================================================================
# EFFECT HANDLERS
# ============================================================================

class EffectHandler(ABC):
    """
    Base class for effect handlers.
    
    Handlers interpret effects and provide implementations.
    """
    
    @abstractmethod
    def can_handle(self, effect: Effect) -> bool:
        """Check if this handler can handle the effect"""
        pass
    
    @abstractmethod
    def handle(self, effect: Effect) -> Any:
        """Handle the effect and return result"""
        pass


class TimeHandler(EffectHandler):
    """Handler for time-related effects"""
    
    def __init__(self, mode: str = "real"):
        """
        Initialize time handler.
        
        Args:
            mode: "real" for actual time, "mock" for controllable time
        """
        self.mode = mode
        self.mock_time = 0.0
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, (AskCurrentTime, Sleep))
    
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, AskCurrentTime):
            if self.mode == "real":
                return time.time()
            else:
                return self.mock_time
        
        elif isinstance(effect, Sleep):
            if self.mode == "real":
                time.sleep(effect.duration)
            else:
                # In mock mode, just advance mock time
                self.mock_time += effect.duration
            return None
    
    def set_mock_time(self, t: float):
        """Set mock time (for testing)"""
        self.mock_time = t
    
    def advance_time(self, delta: float):
        """Advance mock time (for testing)"""
        self.mock_time += delta


class LogHandler(EffectHandler):
    """Handler for logging effects"""
    
    def __init__(self, mode: str = "print"):
        """
        Initialize log handler.
        
        Args:
            mode: "print" for actual logging, "capture" for testing
        """
        self.mode = mode
        self.captured_logs: List[tuple[str, str]] = []
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, Log)
    
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, Log):
            if self.mode == "print":
                print(f"[{effect.level}] {effect.message}")
            else:
                self.captured_logs.append((effect.level, effect.message))
        return None
    
    def get_logs(self) -> List[tuple[str, str]]:
        """Get captured logs (for testing)"""
        return self.captured_logs.copy()
    
    def clear_logs(self):
        """Clear captured logs"""
        self.captured_logs.clear()


class StateHandler(EffectHandler):
    """Handler for state effects"""
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, (GetState, SetState))
    
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, GetState):
            with self._lock:
                return self.state.get(effect.key)
        
        elif isinstance(effect, SetState):
            with self._lock:
                self.state[effect.key] = effect.value
            return None


class NetworkHandler(EffectHandler):
    """Handler for network effects"""
    
    def __init__(self, mode: str = "real"):
        """
        Initialize network handler.
        
        Args:
            mode: "real" for actual requests, "mock" for testing
        """
        self.mode = mode
        self.mock_responses: Dict[str, Any] = {}
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, NetworkRequest)
    
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, NetworkRequest):
            if self.mode == "real":
                # In real mode, would make actual request
                # For now, just return mock
                raise NotImplementedError("Real network requests not implemented")
            else:
                # Return mock response
                key = f"{effect.method}:{effect.url}"
                return self.mock_responses.get(key, None)
    
    def set_mock_response(self, url: str, response: Any, method: str = "GET"):
        """Set mock response for URL (for testing)"""
        key = f"{method}:{url}"
        self.mock_responses[key] = response


class RandomHandler(EffectHandler):
    """Handler for random value effects"""
    
    def __init__(self, mode: str = "real"):
        """
        Initialize random handler.
        
        Args:
            mode: "real" for actual random, "deterministic" for testing
        """
        self.mode = mode
        self.deterministic_values: List[float] = []
        self.deterministic_index = 0
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, RandomValue)
    
    def handle(self, effect: Effect) -> Any:
        if isinstance(effect, RandomValue):
            if self.mode == "real":
                import random
                return random.uniform(effect.min_val, effect.max_val)
            else:
                # Return deterministic value
                if self.deterministic_index < len(self.deterministic_values):
                    val = self.deterministic_values[self.deterministic_index]
                    self.deterministic_index += 1
                    return val
                return effect.min_val
    
    def set_deterministic_values(self, values: List[float]):
        """Set deterministic values (for testing)"""
        self.deterministic_values = values
        self.deterministic_index = 0


# ============================================================================
# EFFECT SYSTEM
# ============================================================================

class EffectSystem:
    """
    Main effect system that manages handlers.
    
    This is thread-local to allow different handlers in different contexts.
    """
    
    _thread_local = threading.local()
    
    @classmethod
    def _get_handlers(cls) -> List[EffectHandler]:
        """Get handlers for current thread"""
        if not hasattr(cls._thread_local, 'handlers'):
            cls._thread_local.handlers = []
        return cls._thread_local.handlers
    
    @classmethod
    def register_handler(cls, handler: EffectHandler):
        """Register an effect handler"""
        handlers = cls._get_handlers()
        handlers.append(handler)
    
    @classmethod
    def unregister_handler(cls, handler: EffectHandler):
        """Unregister an effect handler"""
        handlers = cls._get_handlers()
        if handler in handlers:
            handlers.remove(handler)
    
    @classmethod
    def clear_handlers(cls):
        """Clear all handlers"""
        cls._thread_local.handlers = []
    
    @classmethod
    def perform(cls, effect: Effect) -> Any:
        """
        Perform an effect.
        
        Finds appropriate handler and delegates to it.
        
        Args:
            effect: The effect to perform
            
        Returns:
            Result from the handler
            
        Raises:
            RuntimeError: If no handler can handle the effect
        """
        handlers = cls._get_handlers()
        
        for handler in reversed(handlers):  # Check most recent first
            if handler.can_handle(effect):
                return handler.handle(effect)
        
        raise RuntimeError(
            f"No handler registered for effect: {effect.effect_name()}"
        )


@contextmanager
def with_handlers(*handlers: EffectHandler):
    """
    Context manager for temporarily installing handlers.
    
    Example:
        with with_handlers(TimeHandler("mock"), LogHandler("capture")):
            # Code here uses mock time and captured logging
            result = some_computation()
    """
    # Register handlers
    for handler in handlers:
        EffectSystem.register_handler(handler)
    
    try:
        yield handlers
    finally:
        # Unregister handlers
        for handler in handlers:
            EffectSystem.unregister_handler(handler)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def ask_current_time() -> float:
    """Get current time via effect system"""
    return EffectSystem.perform(AskCurrentTime())


def sleep_effect(duration: float):
    """Sleep via effect system"""
    EffectSystem.perform(Sleep(duration))


def log_effect(level: str, message: str):
    """Log via effect system"""
    EffectSystem.perform(Log(level, message))


def get_state(key: str) -> Any:
    """Get state via effect system"""
    return EffectSystem.perform(GetState(key))


def set_state(key: str, value: Any):
    """Set state via effect system"""
    EffectSystem.perform(SetState(key, value))


def network_request(url: str, method: str = "GET", data: Optional[Any] = None) -> Any:
    """Make network request via effect system"""
    return EffectSystem.perform(NetworkRequest(url, method, data))


def random_value(min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Get random value via effect system"""
    return EffectSystem.perform(RandomValue(min_val, max_val))


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def _example_computation():
    """Example computation using effects"""
    log_effect("INFO", "Starting computation")
    
    start_time = ask_current_time()
    log_effect("INFO", f"Start time: {start_time}")
    
    # Do some work
    sleep_effect(1.0)
    
    end_time = ask_current_time()
    log_effect("INFO", f"End time: {end_time}")
    
    duration = end_time - start_time
    log_effect("INFO", f"Duration: {duration}")
    
    return duration


def _example_usage():
    """Demonstrate algebraic effects"""
    
    print("Algebraic Effects System\n" + "="*70)
    
    print("\n1. Production mode (real time, real logging):")
    with with_handlers(TimeHandler("real"), LogHandler("print")):
        result = _example_computation()
        print(f"   Result: {result:.2f} seconds")
    
    print("\n2. Testing mode (mock time, captured logging):")
    time_handler = TimeHandler("mock")
    log_handler = LogHandler("capture")
    
    time_handler.set_mock_time(1000.0)
    
    with with_handlers(time_handler, log_handler):
        result = _example_computation()
        print(f"   Result: {result:.2f} seconds")
        
        logs = log_handler.get_logs()
        print(f"   Captured {len(logs)} log messages:")
        for level, msg in logs:
            print(f"     [{level}] {msg}")
    
    print("\n3. Deterministic testing:")
    random_handler = RandomHandler("deterministic")
    random_handler.set_deterministic_values([0.5, 0.7, 0.3])
    
    with with_handlers(random_handler):
        values = [random_value() for _ in range(3)]
        print(f"   Deterministic random values: {values}")
    
    print("\n4. State management:")
    state_handler = StateHandler()
    
    with with_handlers(state_handler):
        set_state("counter", 0)
        print(f"   Initial counter: {get_state('counter')}")
        
        set_state("counter", 42)
        print(f"   Updated counter: {get_state('counter')}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    _example_usage()
