"""
Tool registry and lifecycle management.

This module provides the central registry for tools and manages their lifecycle.
"""

from typing import Dict, List, Optional, Set, Callable
import threading
import time

from .base import Tool, ToolConfig, ToolHandle, ToolState, ToolMetrics


class ToolRegistrationError(Exception):
    """Raised when tool registration fails"""
    pass


class ToolNotFoundError(Exception):
    """Raised when tool is not found in registry"""
    pass


class ToolLifecycleError(Exception):
    """Raised when tool lifecycle operation fails"""
    pass


class ToolRegistry:
    """
    Central registry for all tools in the system.
    
    The registry manages tool lifecycle, provides discovery capabilities,
    and coordinates tool interactions.
    """
    
    def __init__(self):
        """Initialize the tool registry"""
        self._tools: Dict[str, ToolHandle] = {}
        self._tools_by_category: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        self._lifecycle_manager = ToolLifecycleManager(self)
    
    def register_tool(self, tool: Tool, config: Optional[ToolConfig] = None) -> ToolHandle:
        """
        Register a tool with the registry.
        
        Args:
            tool: Tool instance to register
            config: Optional config override
            
        Returns:
            ToolHandle for managing the tool
            
        Raises:
            ToolRegistrationError: If registration fails
        """
        with self._lock:
            # Use provided config or tool's config
            cfg = config or tool.config
            tool_id = cfg.tool_id
            
            # Check if already registered
            if tool_id in self._tools:
                raise ToolRegistrationError(f"Tool {tool_id} already registered")
            
            # Create handle
            handle = ToolHandle(tool_id, tool)
            tool._set_handle(handle)
            
            # Register
            self._tools[tool_id] = handle
            
            # Register by category
            category = cfg.tool_category
            if category not in self._tools_by_category:
                self._tools_by_category[category] = set()
            self._tools_by_category[category].add(tool_id)
            
            return handle
    
    def unregister_tool(self, tool_id: str) -> None:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_id: ID of tool to unregister
            
        Raises:
            ToolNotFoundError: If tool not found
        """
        with self._lock:
            if tool_id not in self._tools:
                raise ToolNotFoundError(f"Tool {tool_id} not found")
            
            handle = self._tools[tool_id]
            
            # Ensure tool is terminated
            if handle.state != ToolState.TERMINATED:
                self._lifecycle_manager.detach_tool(tool_id)
            
            # Unregister
            del self._tools[tool_id]
            
            # Unregister from category
            for category_tools in self._tools_by_category.values():
                category_tools.discard(tool_id)
    
    def get_tool(self, tool_id: str) -> Optional[ToolHandle]:
        """
        Get a tool handle by ID.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            ToolHandle if found, None otherwise
        """
        with self._lock:
            return self._tools.get(tool_id)
    
    def list_tools(self, category: Optional[str] = None, 
                   state: Optional[ToolState] = None) -> List[ToolHandle]:
        """
        List tools in the registry.
        
        Args:
            category: Optional category filter
            state: Optional state filter
            
        Returns:
            List of tool handles matching filters
        """
        with self._lock:
            handles = []
            
            # Get tools by category if specified
            if category:
                tool_ids = self._tools_by_category.get(category, set())
                handles = [self._tools[tid] for tid in tool_ids]
            else:
                handles = list(self._tools.values())
            
            # Filter by state if specified
            if state:
                handles = [h for h in handles if h.state == state]
            
            return handles
    
    def find_tools(self, predicate: Callable[[ToolHandle], bool]) -> List[ToolHandle]:
        """
        Find tools matching a predicate.
        
        Args:
            predicate: Function that returns True for matching tools
            
        Returns:
            List of matching tool handles
        """
        with self._lock:
            return [h for h in self._tools.values() if predicate(h)]
    
    @property
    def lifecycle_manager(self) -> 'ToolLifecycleManager':
        """Get lifecycle manager"""
        return self._lifecycle_manager


class ToolLifecycleManager:
    """
    Manages tool lifecycle transitions.
    
    Handles state transitions, initialization, shutdown, suspension, and resumption.
    """
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize lifecycle manager.
        
        Args:
            registry: Tool registry
        """
        self._registry = registry
        self._lock = threading.RLock()
    
    def attach_tool(self, tool_id: str) -> bool:
        """
        Attach a tool (transition DORMANT -> INITIALIZING -> ACTIVE).
        
        Args:
            tool_id: Tool ID
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ToolNotFoundError: If tool not found
            ToolLifecycleError: If attachment fails
        """
        with self._lock:
            handle = self._registry.get_tool(tool_id)
            if not handle:
                raise ToolNotFoundError(f"Tool {tool_id} not found")
            
            tool = handle._tool
            
            # Check current state
            if handle.state != ToolState.DORMANT:
                raise ToolLifecycleError(
                    f"Tool {tool_id} is in state {handle.state}, expected DORMANT"
                )
            
            # Transition to INITIALIZING
            handle.update_state(ToolState.INITIALIZING)
            
            try:
                # Initialize tool
                if not tool.initialize():
                    handle.update_state(ToolState.ERROR)
                    raise ToolLifecycleError(f"Tool {tool_id} initialization failed")
                
                tool._initialized = True
                
                # Transition to ACTIVE
                handle.update_state(ToolState.ACTIVE)
                return True
                
            except Exception as e:
                handle.update_state(ToolState.ERROR)
                handle.record_error(str(e))
                raise ToolLifecycleError(f"Tool {tool_id} attachment failed: {e}")
    
    def detach_tool(self, tool_id: str) -> bool:
        """
        Detach a tool (transition ACTIVE -> TERMINATED).
        
        Args:
            tool_id: Tool ID
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ToolNotFoundError: If tool not found
            ToolLifecycleError: If detachment fails
        """
        with self._lock:
            handle = self._registry.get_tool(tool_id)
            if not handle:
                raise ToolNotFoundError(f"Tool {tool_id} not found")
            
            tool = handle._tool
            
            try:
                # Shutdown tool
                if not tool.shutdown():
                    handle.record_error("Shutdown failed")
                    return False
                
                # Transition to TERMINATED
                handle.update_state(ToolState.TERMINATED)
                return True
                
            except Exception as e:
                handle.update_state(ToolState.ERROR)
                handle.record_error(str(e))
                raise ToolLifecycleError(f"Tool {tool_id} detachment failed: {e}")
    
    def suspend_tool(self, tool_id: str) -> bool:
        """
        Suspend a tool (transition ACTIVE -> SUSPENDED).
        
        Args:
            tool_id: Tool ID
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ToolNotFoundError: If tool not found
            ToolLifecycleError: If suspension fails
        """
        with self._lock:
            handle = self._registry.get_tool(tool_id)
            if not handle:
                raise ToolNotFoundError(f"Tool {tool_id} not found")
            
            tool = handle._tool
            
            # Check current state
            if handle.state != ToolState.ACTIVE:
                raise ToolLifecycleError(
                    f"Tool {tool_id} is in state {handle.state}, expected ACTIVE"
                )
            
            try:
                # Suspend tool
                if not tool.suspend():
                    handle.record_error("Suspension failed")
                    return False
                
                # Transition to SUSPENDED
                handle.update_state(ToolState.SUSPENDED)
                return True
                
            except Exception as e:
                handle.update_state(ToolState.ERROR)
                handle.record_error(str(e))
                raise ToolLifecycleError(f"Tool {tool_id} suspension failed: {e}")
    
    def resume_tool(self, tool_id: str) -> bool:
        """
        Resume a tool (transition SUSPENDED -> ACTIVE).
        
        Args:
            tool_id: Tool ID
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ToolNotFoundError: If tool not found
            ToolLifecycleError: If resumption fails
        """
        with self._lock:
            handle = self._registry.get_tool(tool_id)
            if not handle:
                raise ToolNotFoundError(f"Tool {tool_id} not found")
            
            tool = handle._tool
            
            # Check current state
            if handle.state != ToolState.SUSPENDED:
                raise ToolLifecycleError(
                    f"Tool {tool_id} is in state {handle.state}, expected SUSPENDED"
                )
            
            try:
                # Resume tool
                if not tool.resume():
                    handle.record_error("Resumption failed")
                    return False
                
                # Transition to ACTIVE
                handle.update_state(ToolState.ACTIVE)
                return True
                
            except Exception as e:
                handle.update_state(ToolState.ERROR)
                handle.record_error(str(e))
                raise ToolLifecycleError(f"Tool {tool_id} resumption failed: {e}")
    
    def force_detach_tool(self, tool_id: str) -> bool:
        """
        Forcibly detach a misbehaving tool.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            handle = self._registry.get_tool(tool_id)
            if not handle:
                return False
            
            # Attempt graceful shutdown first
            try:
                tool = handle._tool
                tool.shutdown()
            except Exception:
                pass  # Ignore errors during forced shutdown
            
            # Force transition to TERMINATED
            handle.update_state(ToolState.TERMINATED)
            return True
    
    def get_tool_state(self, tool_id: str) -> Optional[ToolState]:
        """
        Get current tool state.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            ToolState if found, None otherwise
        """
        handle = self._registry.get_tool(tool_id)
        return handle.state if handle else None
    
    def get_tool_metrics(self, tool_id: str) -> Optional[ToolMetrics]:
        """
        Get tool performance metrics.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            ToolMetrics if found, None otherwise
        """
        handle = self._registry.get_tool(tool_id)
        return handle.metrics if handle else None
