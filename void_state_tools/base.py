"""
Base classes and interfaces for Void-State tools.

This module defines the core abstractions that all tools must implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import time
import uuid


class ToolState(Enum):
    """Possible states of a tool"""
    DORMANT = "dormant"           # Registered but not active
    INITIALIZING = "initializing"  # Loading and preparing
    ACTIVE = "active"              # Running and processing
    SUSPENDED = "suspended"        # Temporarily paused
    TERMINATED = "terminated"      # Cleanly shut down
    ERROR = "error"                # Error state


@dataclass
class ToolConfig:
    """Configuration for a tool"""
    # Identity
    tool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    tool_category: str = ""
    version: str = "1.0.0"
    
    # Resource quotas
    max_memory_mb: int = 100
    max_cpu_percent: int = 10
    max_io_ops_per_sec: int = 1000
    max_hooks: int = 10
    max_threads: int = 4
    overhead_budget_ns: int = 1000
    
    # Behavior
    priority: int = 0  # Higher = more important
    enabled: bool = True
    auto_start: bool = True
    
    # Configuration parameters (tool-specific)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetrics:
    """Performance metrics for a tool"""
    # Timing
    total_runtime_seconds: float = 0.0
    average_hook_latency_ns: float = 0.0
    max_hook_latency_ns: float = 0.0
    
    # Resource usage
    current_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    io_operations: int = 0
    
    # Hook statistics
    hooks_registered: int = 0
    hooks_executed: int = 0
    hooks_failed: int = 0
    
    # Event statistics
    events_processed: int = 0
    events_published: int = 0
    
    # Errors
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None


class ToolHandle:
    """Handle for managing a tool instance"""
    
    def __init__(self, tool_id: str, tool: 'Tool'):
        self.tool_id = tool_id
        self._tool = tool
        self._state = ToolState.DORMANT
        self._metrics = ToolMetrics()
        self._start_time = time.time()
    
    @property
    def state(self) -> ToolState:
        """Get current tool state"""
        return self._state
    
    @property
    def metrics(self) -> ToolMetrics:
        """Get tool metrics"""
        self._metrics.total_runtime_seconds = time.time() - self._start_time
        return self._metrics
    
    def update_state(self, new_state: ToolState) -> None:
        """Update tool state"""
        self._state = new_state
    
    def record_hook_execution(self, latency_ns: float, success: bool) -> None:
        """Record hook execution statistics"""
        self._metrics.hooks_executed += 1
        if not success:
            self._metrics.hooks_failed += 1
        
        # Update latency statistics
        if self._metrics.hooks_executed == 1:
            self._metrics.average_hook_latency_ns = latency_ns
        else:
            # Running average
            self._metrics.average_hook_latency_ns = (
                (self._metrics.average_hook_latency_ns * (self._metrics.hooks_executed - 1) + latency_ns)
                / self._metrics.hooks_executed
            )
        
        self._metrics.max_hook_latency_ns = max(self._metrics.max_hook_latency_ns, latency_ns)
    
    def record_error(self, error: str) -> None:
        """Record an error"""
        self._metrics.error_count += 1
        self._metrics.last_error = error
        self._metrics.last_error_time = time.time()


class Tool(ABC):
    """
    Abstract base class for all Void-State tools.
    
    All tools must inherit from this class and implement the required methods.
    Tools are the fundamental building blocks of the Void-State system's
    introspection, maintenance, education, mutation, and defense capabilities.
    """
    
    def __init__(self, config: ToolConfig):
        """
        Initialize the tool.
        
        Args:
            config: Tool configuration
        """
        self.config = config
        self._initialized = False
        self._handle: Optional[ToolHandle] = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the tool.
        
        Called when the tool transitions from DORMANT to INITIALIZING.
        Perform any setup, resource allocation, or preparation here.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """
        Shutdown the tool.
        
        Called when the tool transitions to TERMINATED.
        Clean up resources, unregister hooks, and perform finalization.
        
        Returns:
            True if shutdown succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def suspend(self) -> bool:
        """
        Suspend the tool.
        
        Called when the tool transitions to SUSPENDED.
        Pause processing but maintain state for later resumption.
        
        Returns:
            True if suspension succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def resume(self) -> bool:
        """
        Resume the tool.
        
        Called when the tool transitions from SUSPENDED to ACTIVE.
        Resume processing from suspended state.
        
        Returns:
            True if resumption succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get tool metadata.
        
        Returns:
            Dictionary containing tool metadata including:
                - name: Tool name
                - category: Tool category
                - version: Tool version
                - description: Tool description
                - capabilities: Set of capabilities required
                - dependencies: Set of tool dependencies
        """
        pass
    
    @property
    def tool_id(self) -> str:
        """Get unique tool identifier"""
        return self.config.tool_id
    
    @property
    def is_initialized(self) -> bool:
        """Check if tool is initialized"""
        return self._initialized
    
    def _set_handle(self, handle: ToolHandle) -> None:
        """Set tool handle (internal use only)"""
        self._handle = handle
    
    def get_handle(self) -> Optional[ToolHandle]:
        """Get tool handle"""
        return self._handle


class AnalysisTool(Tool):
    """
    Base class for analysis tools.
    
    Analysis tools observe system state and produce insights without
    modifying system behavior.
    """
    
    @abstractmethod
    def analyze(self, data: Any) -> Any:
        """
        Analyze input data and produce results.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Analysis results
        """
        pass


class InterceptorTool(Tool):
    """
    Base class for interceptor tools.
    
    Interceptor tools can observe and modify system operations.
    """
    
    @abstractmethod
    def intercept(self, operation: Any, context: Any) -> Any:
        """
        Intercept an operation.
        
        Args:
            operation: The operation to intercept
            context: Execution context
            
        Returns:
            Intercept decision (ALLOW, DENY, MODIFY, etc.)
        """
        pass


class MonitoringTool(Tool):
    """
    Base class for monitoring tools.
    
    Monitoring tools continuously observe system behavior and metrics.
    """
    
    @abstractmethod
    def on_event(self, event: Any) -> None:
        """
        Handle a monitoring event.
        
        Args:
            event: The event to handle
        """
        pass


class SynthesisTool(Tool):
    """
    Base class for synthesis tools.
    
    Synthesis tools generate new artifacts (code, data, tools, etc.)
    from specifications or other inputs.
    """
    
    @abstractmethod
    def synthesize(self, specification: Any) -> Any:
        """
        Synthesize new artifact from specification.
        
        Args:
            specification: Specification of what to synthesize
            
        Returns:
            Synthesized artifact
        """
        pass


# Type aliases for common data structures
MemoryState = bytes
State = Dict[str, Any]
Event = Dict[str, Any]
Snapshot = Dict[str, Any]
Pattern = Any
