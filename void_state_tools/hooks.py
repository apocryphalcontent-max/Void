"""
Hook system for VM and Kernel integration.

This module provides the hook infrastructure that allows tools to attach
to various points in the VM and Kernel.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass
import time


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
    additional_data: Dict[str, Any]


class HookPoint:
    """
    Represents a hook point where tools can attach.
    
    Hook points are locations in the VM or Kernel where tools can
    register callbacks to be executed.
    """
    
    def __init__(self, name: str, timing: HookTiming, overhead_budget_ns: int = 1000):
        """
        Initialize hook point.
        
        Args:
            name: Hook point name
            timing: When hooks execute (before/after/around)
            overhead_budget_ns: Overhead budget in nanoseconds
        """
        self.name = name
        self.timing = timing
        self.overhead_budget_ns = overhead_budget_ns
        self._callbacks: List[tuple] = []  # (priority, callback, filter)
    
    def register(self, 
                callback: Callable,
                priority: int = HookPriority.NORMAL.value,
                filter_fn: Optional[Callable[[HookContext], bool]] = None) -> int:
        """
        Register a callback at this hook point.
        
        Args:
            callback: Function to call
            priority: Execution priority (higher = earlier)
            filter_fn: Optional filter to determine if hook should fire
            
        Returns:
            Registration ID for later deregistration
        """
        registration_id = id(callback)
        self._callbacks.append((priority, callback, filter_fn, registration_id))
        # Sort by priority (descending)
        self._callbacks.sort(key=lambda x: x[0], reverse=True)
        return registration_id
    
    def unregister(self, registration_id: int) -> bool:
        """
        Unregister a callback.
        
        Args:
            registration_id: ID returned from register()
            
        Returns:
            True if successful, False if not found
        """
        for i, (_, _, _, reg_id) in enumerate(self._callbacks):
            if reg_id == registration_id:
                del self._callbacks[i]
                return True
        return False
    
    def execute(self, context: HookContext, *args, **kwargs) -> List[Any]:
        """
        Execute all registered callbacks.
        
        Args:
            context: Hook execution context
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
            
        Returns:
            List of results from callbacks
        """
        results = []
        
        for priority, callback, filter_fn, _ in self._callbacks:
            # Check filter
            if filter_fn and not filter_fn(context):
                continue
            
            # Execute callback with timing
            start = time.perf_counter_ns()
            try:
                result = callback(context, *args, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error but don't propagate
                print(f"Hook callback error: {e}")
                results.append(None)
            
            # Check overhead budget
            elapsed = time.perf_counter_ns() - start
            if elapsed > self.overhead_budget_ns:
                print(f"Warning: Hook {self.name} exceeded budget: {elapsed}ns > {self.overhead_budget_ns}ns")
        
        return results


class VMHooks:
    """
    Collection of VM hook points.
    
    Provides hook points for various VM events and operations.
    """
    
    def __init__(self):
        """Initialize VM hooks"""
        # Per-cycle hooks
        self.before_cycle = HookPoint("vm.before_cycle", HookTiming.BEFORE, overhead_budget_ns=100)
        self.after_cycle = HookPoint("vm.after_cycle", HookTiming.AFTER, overhead_budget_ns=100)
        
        # Per-instruction hooks (by opcode)
        self.before_instruction: Dict[str, HookPoint] = {}
        self.after_instruction: Dict[str, HookPoint] = {}
        
        # Event hooks
        self.on_exception = HookPoint("vm.on_exception", HookTiming.BEFORE, overhead_budget_ns=1000)
        self.on_io = HookPoint("vm.on_io", HookTiming.BEFORE, overhead_budget_ns=1000)
        self.on_memory_access = HookPoint("vm.on_memory_access", HookTiming.BEFORE, overhead_budget_ns=50)
        
        # Snapshot hooks
        self.before_snapshot = HookPoint("vm.before_snapshot", HookTiming.BEFORE, overhead_budget_ns=10_000_000)
        self.after_snapshot = HookPoint("vm.after_snapshot", HookTiming.AFTER, overhead_budget_ns=10_000_000)
    
    def get_instruction_hook(self, opcode: str, timing: HookTiming) -> HookPoint:
        """
        Get or create a hook point for a specific instruction.
        
        Args:
            opcode: Instruction opcode
            timing: Before or after instruction
            
        Returns:
            HookPoint for that instruction
        """
        hooks_dict = self.before_instruction if timing == HookTiming.BEFORE else self.after_instruction
        
        if opcode not in hooks_dict:
            name = f"vm.{timing.value}_instruction.{opcode}"
            hooks_dict[opcode] = HookPoint(name, timing, overhead_budget_ns=50)
        
        return hooks_dict[opcode]


class KernelHooks:
    """
    Collection of Kernel hook points.
    
    Provides hook points for various Kernel events and operations.
    """
    
    def __init__(self):
        """Initialize Kernel hooks"""
        # System call hooks
        self.syscall_intercept: Dict[str, HookPoint] = {}
        
        # Memory management hooks
        self.on_memory_allocation = HookPoint("kernel.on_alloc", HookTiming.BEFORE, overhead_budget_ns=500)
        self.on_memory_deallocation = HookPoint("kernel.on_dealloc", HookTiming.AFTER, overhead_budget_ns=500)
        self.on_page_fault = HookPoint("kernel.on_page_fault", HookTiming.BEFORE, overhead_budget_ns=10_000)
        self.on_gc_cycle = HookPoint("kernel.on_gc", HookTiming.AROUND, overhead_budget_ns=1_000_000)
        
        # Scheduler hooks
        self.on_task_schedule = HookPoint("kernel.on_schedule", HookTiming.BEFORE, overhead_budget_ns=1000)
        self.on_context_switch = HookPoint("kernel.on_context_switch", HookTiming.AROUND, overhead_budget_ns=500)
        self.on_task_complete = HookPoint("kernel.on_task_complete", HookTiming.AFTER, overhead_budget_ns=1000)
        
        # I/O hooks
        self.on_io_request = HookPoint("kernel.on_io_request", HookTiming.BEFORE, overhead_budget_ns=1000)
        self.on_device_io = HookPoint("kernel.on_device_io", HookTiming.BEFORE, overhead_budget_ns=1000)
    
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
            self.syscall_intercept[syscall] = HookPoint(name, HookTiming.BEFORE, overhead_budget_ns=1000)
        
        return self.syscall_intercept[syscall]


class HookRegistry:
    """
    Central registry for all hook points.
    
    Manages VM and Kernel hooks and provides unified access.
    """
    
    def __init__(self):
        """Initialize hook registry"""
        self.vm_hooks = VMHooks()
        self.kernel_hooks = KernelHooks()
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


class HookFilter(ABC):
    """Base class for hook filters"""
    
    @abstractmethod
    def should_trigger(self, context: HookContext) -> bool:
        """
        Determine if hook should trigger.
        
        Args:
            context: Hook context
            
        Returns:
            True if hook should trigger, False otherwise
        """
        pass


class FrequencyFilter(HookFilter):
    """Filter that samples at a specified frequency"""
    
    def __init__(self, sample_rate: float = 0.01):
        """
        Initialize frequency filter.
        
        Args:
            sample_rate: Fraction of events to sample [0.0, 1.0]
        """
        self.sample_rate = sample_rate
        self.counter = 0
        self.interval = int(1.0 / sample_rate) if sample_rate > 0 else 0
    
    def should_trigger(self, context: HookContext) -> bool:
        """Sample at specified frequency"""
        if self.interval == 0:
            return False
        self.counter += 1
        return (self.counter % self.interval) == 0


class TimeWindowFilter(HookFilter):
    """Filter that triggers only within a time window"""
    
    def __init__(self, start_time: float, end_time: float):
        """
        Initialize time window filter.
        
        Args:
            start_time: Start of window (timestamp)
            end_time: End of window (timestamp)
        """
        self.start_time = start_time
        self.end_time = end_time
    
    def should_trigger(self, context: HookContext) -> bool:
        """Trigger only within time window"""
        return self.start_time <= context.timestamp <= self.end_time


class ConditionalFilter(HookFilter):
    """Filter based on custom condition"""
    
    def __init__(self, condition: Callable[[HookContext], bool]):
        """
        Initialize conditional filter.
        
        Args:
            condition: Function that returns True if hook should trigger
        """
        self.condition = condition
    
    def should_trigger(self, context: HookContext) -> bool:
        """Apply custom condition"""
        return self.condition(context)
