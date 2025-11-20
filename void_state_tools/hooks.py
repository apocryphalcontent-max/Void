"""
Enhanced Hook system with overhead enforcement and automatic detachment.

This module provides sophisticated hook infrastructure with:
- Per-callback violation tracking
- Automatic detachment after consecutive violations
- Clock injection for deterministic testing
- Thread-safe operation
"""

import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .clock import Clock, get_clock


class HookTiming(Enum):
    """When a hook executes relative to an event"""
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"


class HookPriority(Enum):
    """Hook execution priority"""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 0


@dataclass
class HookContext:
    """Context information for a hook execution"""
    timestamp: float
    cycle_count: int
    thread_id: int
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackRegistration:
    """
    Registration record for a callback with violation tracking.

    Sophisticated structure that tracks:
    - Callback metadata
    - Violation history
    - Performance statistics
    """
    callback_id: int
    callback: Callable
    priority: int
    filter_fn: Optional[Callable[[HookContext], bool]]

    # Violation tracking
    consecutive_violations: int = 0
    total_violations: int = 0
    total_executions: int = 0
    total_latency_ns: int = 0
    max_latency_ns: int = 0

    # State
    enabled: bool = True
    detachment_reason: Optional[str] = None


class HookPoint:
    """
    Enhanced hook point with automatic enforcement.

    Features:
    - Per-callback violation tracking
    - Automatic detachment after threshold violations
    - Performance monitoring
    - Clock injection for testing
    """

    def __init__(
        self,
        name: str,
        timing: HookTiming,
        overhead_budget_ns: int = 1000,
        violation_threshold: int = 3,
        clock: Optional[Clock] = None
    ):
        """
        Initialize hook point.

        Args:
            name: Hook point name
            timing: When hooks execute (before/after/around)
            overhead_budget_ns: Overhead budget in nanoseconds
            violation_threshold: Number of consecutive violations before detachment
            clock: Optional clock instance (defaults to system clock)
        """
        self.name = name
        self.timing = timing
        self.overhead_budget_ns = overhead_budget_ns
        self.violation_threshold = violation_threshold
        self._clock = clock or get_clock()

        self._registrations: List[CallbackRegistration] = []
        self._lock = threading.RLock()
        self._execution_count = 0

    def register(
        self,
        callback: Callable,
        priority: int = HookPriority.NORMAL.value,
        filter_fn: Optional[Callable[[HookContext], bool]] = None
    ) -> int:
        """
        Register a callback at this hook point.

        Args:
            callback: Function to call
            priority: Execution priority (higher = earlier)
            filter_fn: Optional filter to determine if hook should fire

        Returns:
            Registration ID for later deregistration
        """
        with self._lock:
            callback_id = id(callback)

            reg = CallbackRegistration(
                callback_id=callback_id,
                callback=callback,
                priority=priority,
                filter_fn=filter_fn
            )

            self._registrations.append(reg)

            # Sort by priority (descending)
            self._registrations.sort(key=lambda r: r.priority, reverse=True)

            return callback_id

    def unregister(self, callback_id: int) -> bool:
        """
        Unregister a callback.

        Args:
            callback_id: ID returned from register()

        Returns:
            True if successful, False if not found
        """
        with self._lock:
            for i, reg in enumerate(self._registrations):
                if reg.callback_id == callback_id:
                    del self._registrations[i]
                    return True
            return False

    def execute(self, context: HookContext, *args, **kwargs) -> List[Any]:
        """
        Execute all registered callbacks with enforcement.

        Tracks per-callback overhead and automatically detaches callbacks
        that repeatedly violate the budget.

        Args:
            context: Hook execution context
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks

        Returns:
            List of results from callbacks
        """
        with self._lock:
            self._execution_count += 1
            results = []
            detached_callbacks = []

            for reg in self._registrations:
                # Skip if disabled
                if not reg.enabled:
                    continue

                # Check filter
                if reg.filter_fn and not reg.filter_fn(context):
                    continue

                # Execute callback with timing
                start_ns = self._clock.now_ns()

                try:
                    result = reg.callback(context, *args, **kwargs)
                    results.append(result)
                    success = True

                except Exception:
                    # Log error but don't propagate
                    results.append(None)
                    success = False
                    # Violations count for both exceptions and overhead

                # Measure latency
                end_ns = self._clock.now_ns()
                latency_ns = end_ns - start_ns

                # Update statistics
                reg.total_executions += 1
                reg.total_latency_ns += latency_ns
                reg.max_latency_ns = max(reg.max_latency_ns, latency_ns)

                # Check overhead budget
                if latency_ns > self.overhead_budget_ns or not success:
                    reg.consecutive_violations += 1
                    reg.total_violations += 1

                    # Check if threshold exceeded
                    if reg.consecutive_violations >= self.violation_threshold:
                        reg.enabled = False
                        reg.detachment_reason = (
                            f"Exceeded overhead budget {reg.consecutive_violations} "
                            f"consecutive times (budget: {self.overhead_budget_ns}ns, "
                            f"max latency: {reg.max_latency_ns}ns)"
                        )
                        detached_callbacks.append(reg)
                else:
                    # Reset consecutive violations on successful execution
                    reg.consecutive_violations = 0

            # Report detached callbacks
            for reg in detached_callbacks:
                print(
                    f"⚠️  Hook {self.name}: Forcibly detached callback {reg.callback_id} "
                    f"- {reg.detachment_reason}"
                )

            return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get hook point statistics.

        Returns:
            dict: Statistics including execution count, active callbacks, violations
        """
        with self._lock:
            active_callbacks = sum(1 for r in self._registrations if r.enabled)
            total_callbacks = len(self._registrations)
            detached_callbacks = total_callbacks - active_callbacks

            total_violations = sum(r.total_violations for r in self._registrations)

            return {
                "name": self.name,
                "execution_count": self._execution_count,
                "overhead_budget_ns": self.overhead_budget_ns,
                "violation_threshold": self.violation_threshold,
                "active_callbacks": active_callbacks,
                "total_callbacks": total_callbacks,
                "detached_callbacks": detached_callbacks,
                "total_violations": total_violations,
            }

    def get_callback_statistics(self) -> List[Dict[str, Any]]:
        """
        Get per-callback statistics.

        Returns:
            list: List of callback statistics
        """
        with self._lock:
            stats = []
            for reg in self._registrations:
                avg_latency_ns = (
                    reg.total_latency_ns / reg.total_executions
                    if reg.total_executions > 0
                    else 0
                )

                stats.append({
                    "callback_id": reg.callback_id,
                    "priority": reg.priority,
                    "enabled": reg.enabled,
                    "total_executions": reg.total_executions,
                    "total_violations": reg.total_violations,
                    "consecutive_violations": reg.consecutive_violations,
                    "avg_latency_ns": avg_latency_ns,
                    "max_latency_ns": reg.max_latency_ns,
                    "detachment_reason": reg.detachment_reason,
                })

            return stats


class VMHooks:
    """
    Collection of VM hook points.

    Provides hook points for various VM events and operations.
    """

    def __init__(self, clock: Optional[Clock] = None):
        """
        Initialize VM hooks.

        Args:
            clock: Optional clock instance for all hook points
        """
        self._clock = clock or get_clock()

        # Per-cycle hooks
        self.before_cycle = HookPoint(
            "vm.before_cycle", HookTiming.BEFORE,
            overhead_budget_ns=100, clock=self._clock
        )
        self.after_cycle = HookPoint(
            "vm.after_cycle", HookTiming.AFTER,
            overhead_budget_ns=100, clock=self._clock
        )

        # Per-instruction hooks (by opcode)
        self.before_instruction: Dict[str, HookPoint] = {}
        self.after_instruction: Dict[str, HookPoint] = {}

        # Event hooks
        self.on_exception = HookPoint(
            "vm.on_exception", HookTiming.BEFORE,
            overhead_budget_ns=1000, clock=self._clock
        )
        self.on_io = HookPoint(
            "vm.on_io", HookTiming.BEFORE,
            overhead_budget_ns=1000, clock=self._clock
        )
        self.on_memory_access = HookPoint(
            "vm.on_memory_access", HookTiming.BEFORE,
            overhead_budget_ns=50, clock=self._clock
        )

        # Snapshot hooks
        self.before_snapshot = HookPoint(
            "vm.before_snapshot", HookTiming.BEFORE,
            overhead_budget_ns=10_000_000, clock=self._clock
        )
        self.after_snapshot = HookPoint(
            "vm.after_snapshot", HookTiming.AFTER,
            overhead_budget_ns=10_000_000, clock=self._clock
        )

    def get_instruction_hook(self, opcode: str, timing: HookTiming) -> HookPoint:
        """
        Get or create a hook point for a specific instruction.

        Args:
            opcode: Instruction opcode
            timing: Before or after instruction

        Returns:
            HookPoint for that instruction
        """
        hooks_dict = (
            self.before_instruction
            if timing == HookTiming.BEFORE
            else self.after_instruction
        )

        if opcode not in hooks_dict:
            name = f"vm.{timing.value}_instruction.{opcode}"
            hooks_dict[opcode] = HookPoint(
                name, timing,
                overhead_budget_ns=50,
                clock=self._clock
            )

        return hooks_dict[opcode]


class KernelHooks:
    """
    Collection of Kernel hook points.

    Provides hook points for various Kernel events and operations.
    """

    def __init__(self, clock: Optional[Clock] = None):
        """
        Initialize Kernel hooks.

        Args:
            clock: Optional clock instance for all hook points
        """
        self._clock = clock or get_clock()

        # System call hooks
        self.syscall_intercept: Dict[str, HookPoint] = {}

        # Memory management hooks
        self.on_memory_allocation = HookPoint(
            "kernel.on_alloc", HookTiming.BEFORE,
            overhead_budget_ns=500, clock=self._clock
        )
        self.on_memory_deallocation = HookPoint(
            "kernel.on_dealloc", HookTiming.AFTER,
            overhead_budget_ns=500, clock=self._clock
        )
        self.on_page_fault = HookPoint(
            "kernel.on_page_fault", HookTiming.BEFORE,
            overhead_budget_ns=10_000, clock=self._clock
        )
        self.on_gc_cycle = HookPoint(
            "kernel.on_gc", HookTiming.AROUND,
            overhead_budget_ns=1_000_000, clock=self._clock
        )

        # Scheduler hooks
        self.on_task_schedule = HookPoint(
            "kernel.on_schedule", HookTiming.BEFORE,
            overhead_budget_ns=1000, clock=self._clock
        )
        self.on_context_switch = HookPoint(
            "kernel.on_context_switch", HookTiming.AROUND,
            overhead_budget_ns=500, clock=self._clock
        )
        self.on_task_complete = HookPoint(
            "kernel.on_task_complete", HookTiming.AFTER,
            overhead_budget_ns=1000, clock=self._clock
        )

        # I/O hooks
        self.on_io_request = HookPoint(
            "kernel.on_io_request", HookTiming.BEFORE,
            overhead_budget_ns=1000, clock=self._clock
        )
        self.on_device_io = HookPoint(
            "kernel.on_device_io", HookTiming.BEFORE,
            overhead_budget_ns=1000, clock=self._clock
        )

    def get_syscall_hook(self, syscall: str) -> HookPoint:
        """
        Get or create a hook point for a specific system call.

        Args:
            syscall: System call name

        Returns:
            HookPoint for that syscall
        """
        if syscall not in self.syscall_intercept:
            name = f"kernel.syscall.{syscall}"
            self.syscall_intercept[syscall] = HookPoint(
                name, HookTiming.BEFORE,
                overhead_budget_ns=1000,
                clock=self._clock
            )

        return self.syscall_intercept[syscall]


class HookRegistry:
    """
    Central registry for all hook points.

    Manages VM and Kernel hooks and provides unified access.
    """

    def __init__(self, clock: Optional[Clock] = None):
        """
        Initialize hook registry.

        Args:
            clock: Optional clock instance for all hooks
        """
        self._clock = clock or get_clock()
        self.vm_hooks = VMHooks(clock=self._clock)
        self.kernel_hooks = KernelHooks(clock=self._clock)
        self._all_hooks: Dict[str, HookPoint] = {}

        # Index all hooks
        self._index_hooks()

    def _index_hooks(self) -> None:
        """Index all available hooks"""
        # Index VM hooks
        for attr_name in dir(self.vm_hooks):
            if not attr_name.startswith('_'):
                attr = getattr(self.vm_hooks, attr_name)
                if isinstance(attr, HookPoint):
                    self._all_hooks[attr.name] = attr

        # Index Kernel hooks
        for attr_name in dir(self.kernel_hooks):
            if not attr_name.startswith('_'):
                attr = getattr(self.kernel_hooks, attr_name)
                if isinstance(attr, HookPoint):
                    self._all_hooks[attr.name] = attr

    def get_hook(self, name: str) -> Optional[HookPoint]:
        """
        Get a hook point by name.

        Args:
            name: Hook point name

        Returns:
            HookPoint if found, None otherwise
        """
        return self._all_hooks.get(name)

    def list_hooks(self, prefix: Optional[str] = None) -> List[str]:
        """
        List available hook points.

        Args:
            prefix: Optional prefix filter (e.g., "vm." or "kernel.")

        Returns:
            List of hook point names
        """
        if prefix:
            return [name for name in self._all_hooks.keys() if name.startswith(prefix)]
        return list(self._all_hooks.keys())

    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get global hook statistics.

        Returns:
            dict: Aggregate statistics across all hooks
        """
        total_executions = 0
        total_violations = 0
        total_detached = 0

        for hook in self._all_hooks.values():
            stats = hook.get_statistics()
            total_executions += stats["execution_count"]
            total_violations += stats["total_violations"]
            total_detached += stats["detached_callbacks"]

        return {
            "total_hooks": len(self._all_hooks),
            "total_executions": total_executions,
            "total_violations": total_violations,
            "total_detached_callbacks": total_detached,
        }


# Hook filters (from original hooks.py)
class HookFilter(ABC):
    """Base class for hook filters"""

    @abstractmethod
    def should_trigger(self, context: HookContext) -> bool:
        """Determine if hook should trigger"""
        pass


class FrequencyFilter(HookFilter):
    """Filter that samples at a specified frequency"""

    def __init__(self, sample_rate: float = 0.01):
        self.sample_rate = sample_rate
        self.counter = 0
        self.interval = int(1.0 / sample_rate) if sample_rate > 0 else 0

    def should_trigger(self, context: HookContext) -> bool:
        if self.interval == 0:
            return False
        self.counter += 1
        return (self.counter % self.interval) == 0


class TimeWindowFilter(HookFilter):
    """Filter that triggers only within a time window"""

    def __init__(self, start_time: float, end_time: float):
        self.start_time = start_time
        self.end_time = end_time

    def should_trigger(self, context: HookContext) -> bool:
        return self.start_time <= context.timestamp <= self.end_time


class ConditionalFilter(HookFilter):
    """Filter based on custom condition"""

    def __init__(self, condition: Callable[[HookContext], bool]):
        self.condition = condition

    def should_trigger(self, context: HookContext) -> bool:
        return self.condition(context)


# Export public API
__all__ = [
    'HookPoint',
    'HookContext',
    'HookTiming',
    'HookPriority',
    'HookRegistry',
    'VMHooks',
    'KernelHooks',
    'HookFilter',
    'FrequencyFilter',
    'TimeWindowFilter',
    'ConditionalFilter',
    'CallbackRegistration',
]
