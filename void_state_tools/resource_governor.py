"""
ResourceGovernor for monitoring and enforcing tool resource quotas.

Provides real-time monitoring of tool resource usage (CPU, memory, I/O)
and automatic enforcement actions (throttle, suspend, terminate) when
quotas are violated.
"""

import time
import threading
from typing import Dict, Optional, Callable, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

if TYPE_CHECKING:
    from .base import ToolConfig

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceViolation(Enum):
    """Types of resource quota violations."""
    CPU_EXCEEDED = "cpu_exceeded"
    MEMORY_EXCEEDED = "memory_exceeded"
    IO_EXCEEDED = "io_exceeded"
    HOOKS_EXCEEDED = "hooks_exceeded"
    THREADS_EXCEEDED = "threads_exceeded"


class EnforcementAction(Enum):
    """Actions taken when quotas are violated."""
    WARN = "warn"
    THROTTLE = "throttle"
    SUSPEND = "suspend"
    TERMINATE = "terminate"


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    timestamp: float
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    io_ops_per_sec: float = 0.0
    active_hooks: int = 0
    active_threads: int = 0


@dataclass
class ViolationRecord:
    """Record of a resource quota violation."""
    timestamp: float
    tool_id: str
    violation_type: ResourceViolation
    actual_value: float
    quota_value: float
    action_taken: EnforcementAction


@dataclass
class QuotaPolicy:
    """Policy for handling quota violations."""
    # Number of violations before escalating action
    warn_threshold: int = 1
    throttle_threshold: int = 2
    suspend_threshold: int = 3
    terminate_threshold: int = 5

    # Enforcement actions enabled
    enable_throttle: bool = True
    enable_suspend: bool = True
    enable_terminate: bool = False  # Default to safe

    # Callbacks
    violation_callback: Optional[Callable[[ViolationRecord], None]] = None


class ResourceGovernor:
    """
    ResourceGovernor monitors and enforces tool resource quotas.

    Features:
    - Real-time CPU/memory monitoring via psutil
    - Configurable quota policies with escalating enforcement
    - Audit trail of all violations and actions
    - Integration with ToolRegistry for automatic enforcement
    """

    def __init__(self, policy: Optional[QuotaPolicy] = None, clock: Optional['Clock'] = None):
        """
        Initialize the ResourceGovernor.

        Args:
            policy: Quota enforcement policy (uses default if None).
            clock: Optional clock instance for time tracking.
        """
        from .clock import get_clock
        self.policy = policy or QuotaPolicy()
        self._clock = clock or get_clock()

        # Tool resource tracking
        self._tool_usage: Dict[str, deque] = {}  # deque of ResourceUsage
        self._tool_quotas: Dict[str, 'ToolConfig'] = {}
        self._violation_counts: Dict[str, Dict[ResourceViolation, int]] = {}
        self._violation_history: deque = deque(maxlen=1000)  # Bounded history

        # Monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 1.0  # seconds
        self._lock = threading.RLock()

        # Process handle
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None

    def start_monitoring(self, interval: float = 1.0):
        """
        Start background resource monitoring.

        Args:
            interval: Monitoring interval in seconds.
        """
        if self._monitoring:
            return

        self._monitor_interval = interval
        self._monitoring = True

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def register_tool(self, tool_id: str, config: 'ToolConfig'):
        """
        Register a tool for resource monitoring.

        Args:
            tool_id: Tool identifier.
            config: Tool configuration with resource quotas.
        """
        with self._lock:
            self._tool_quotas[tool_id] = config
            self._tool_usage[tool_id] = deque(maxlen=100)  # Keep last 100 samples
            self._violation_counts[tool_id] = {v: 0 for v in ResourceViolation}

    def unregister_tool(self, tool_id: str):
        """
        Unregister a tool from monitoring.

        Args:
            tool_id: Tool identifier.
        """
        with self._lock:
            self._tool_quotas.pop(tool_id, None)
            self._tool_usage.pop(tool_id, None)
            self._violation_counts.pop(tool_id, None)

    def record_usage(self, tool_id: str, usage: ResourceUsage):
        """
        Record resource usage for a tool.

        Args:
            tool_id: Tool identifier.
            usage: Resource usage snapshot.
        """
        with self._lock:
            if tool_id not in self._tool_usage:
                return

            self._tool_usage[tool_id].append(usage)
            # deque automatically maintains maxlen, no manual truncation needed

            # Check for violations
            self._check_quotas(tool_id, usage)

    def get_usage(self, tool_id: str) -> List[ResourceUsage]:
        """
        Get resource usage history for a tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            list: List of ResourceUsage snapshots.
        """
        with self._lock:
            return list(self._tool_usage.get(tool_id, []))

    def get_current_usage(self, tool_id: str) -> Optional[ResourceUsage]:
        """
        Get current resource usage for a tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            Optional[ResourceUsage]: Latest usage snapshot, or None if not found.
        """
        with self._lock:
            history = self._tool_usage.get(tool_id, [])
            return history[-1] if history else None

    def get_violations(self, tool_id: Optional[str] = None) -> List[ViolationRecord]:
        """
        Get violation history.

        Args:
            tool_id: Optional tool ID filter.

        Returns:
            list: List of violation records.
        """
        with self._lock:
            if tool_id:
                return [v for v in self._violation_history if v.tool_id == tool_id]
            return list(self._violation_history)

    def _check_quotas(self, tool_id: str, usage: ResourceUsage):
        """
        Check if usage exceeds quotas and take enforcement action.

        Args:
            tool_id: Tool identifier.
            usage: Resource usage snapshot.
        """
        config = self._tool_quotas.get(tool_id)
        if not config:
            return

        violations = []

        # Check CPU quota
        if usage.cpu_percent > config.max_cpu_percent:
            violations.append((
                ResourceViolation.CPU_EXCEEDED,
                usage.cpu_percent,
                config.max_cpu_percent
            ))

        # Check memory quota
        if usage.memory_mb > config.max_memory_mb:
            violations.append((
                ResourceViolation.MEMORY_EXCEEDED,
                usage.memory_mb,
                config.max_memory_mb
            ))

        # Check I/O quota
        if usage.io_ops_per_sec > config.max_io_ops_per_sec:
            violations.append((
                ResourceViolation.IO_EXCEEDED,
                usage.io_ops_per_sec,
                config.max_io_ops_per_sec
            ))

        # Check hooks quota
        if usage.active_hooks > config.max_hooks:
            violations.append((
                ResourceViolation.HOOKS_EXCEEDED,
                usage.active_hooks,
                config.max_hooks
            ))

        # Check threads quota
        if usage.active_threads > config.max_threads:
            violations.append((
                ResourceViolation.THREADS_EXCEEDED,
                usage.active_threads,
                config.max_threads
            ))

        # Handle violations
        for violation_type, actual, quota in violations:
            self._handle_violation(tool_id, violation_type, actual, quota)

    def _handle_violation(
        self,
        tool_id: str,
        violation_type: ResourceViolation,
        actual_value: float,
        quota_value: float
    ):
        """
        Handle a resource quota violation.

        Args:
            tool_id: Tool identifier.
            violation_type: Type of violation.
            actual_value: Actual resource usage.
            quota_value: Quota limit.
        """
        # Increment violation count
        self._violation_counts[tool_id][violation_type] += 1
        count = self._violation_counts[tool_id][violation_type]

        # Determine action based on policy
        action = self._determine_action(count)

        # Record violation
        record = ViolationRecord(
            timestamp=time.time(),
            tool_id=tool_id,
            violation_type=violation_type,
            actual_value=actual_value,
            quota_value=quota_value,
            action_taken=action
        )

        self._violation_history.append(record)
        # deque automatically maintains maxlen, no manual truncation needed

        # Invoke callback
        if self.policy.violation_callback:
            try:
                self.policy.violation_callback(record)
            except Exception:
                pass  # Don't let callback errors break enforcement

        # TODO: Integrate with ToolRegistry for actual enforcement
        # This would require a reference to the registry and lifecycle manager

    def _determine_action(self, violation_count: int) -> EnforcementAction:
        """
        Determine enforcement action based on violation count.

        Args:
            violation_count: Number of violations.

        Returns:
            EnforcementAction: Action to take.
        """
        if violation_count >= self.policy.terminate_threshold and self.policy.enable_terminate:
            return EnforcementAction.TERMINATE

        elif violation_count >= self.policy.suspend_threshold and self.policy.enable_suspend:
            return EnforcementAction.SUSPEND

        elif violation_count >= self.policy.throttle_threshold and self.policy.enable_throttle:
            return EnforcementAction.THROTTLE

        else:
            return EnforcementAction.WARN

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._collect_metrics()
            except Exception:
                pass  # Continue monitoring despite errors

            time.sleep(self._monitor_interval)

    def _collect_metrics(self):
        """Collect metrics for all registered tools."""
        if not PSUTIL_AVAILABLE or not self._process:
            return

        with self._lock:
            for tool_id in list(self._tool_quotas.keys()):
                # For now, record aggregate process metrics
                # TODO: Per-tool metric attribution requires instrumentation
                try:
                    usage = ResourceUsage(
                        timestamp=self._clock.now(),
                        cpu_percent=self._process.cpu_percent(interval=0.1),
                        memory_mb=self._process.memory_info().rss / 1024 / 1024,
                        io_ops_per_sec=0.0,  # TODO: Implement I/O tracking
                        active_hooks=0,  # TODO: Track from hook registry
                        active_threads=threading.active_count()
                    )

                    self.record_usage(tool_id, usage)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get governor statistics.

        Returns:
            dict: Statistics including violation counts, active tools, etc.
        """
        with self._lock:
            return {
                "active_tools": len(self._tool_quotas),
                "total_violations": len(self._violation_history),
                "violations_by_type": {
                    vtype.value: sum(
                        1 for v in self._violation_history
                        if v.violation_type == vtype
                    )
                    for vtype in ResourceViolation
                },
                "monitoring_active": self._monitoring,
                "psutil_available": PSUTIL_AVAILABLE
            }


# Export public API
__all__ = [
    'ResourceGovernor',
    'ResourceUsage',
    'ResourceViolation',
    'EnforcementAction',
    'ViolationRecord',
    'QuotaPolicy'
]
