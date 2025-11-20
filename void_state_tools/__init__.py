"""
Void-State Proprietary Tools: Self-Aware AI Infrastructure

This package provides a comprehensive toolkit for AI system introspection,
enabling deep self-awareness and runtime transparency across 5 architectural layers.

Version: 1.0.0-mvp-complete
"""

__version__ = "1.0.0"
__author__ = "Void State Team"
__license__ = "Proprietary"

# Layer 0: Integration Substrate
from .base import (
    Tool,
    ToolConfig,
    ToolState,
    ToolMetrics,
    ToolHandle,
    AnalysisTool,
    InterceptorTool,
    MonitoringTool,
    SynthesisTool,
)

from .hooks import (
    HookPoint,
    HookContext,
    HookTiming,
    HookPriority,
    HookRegistry,
    VMHooks,
    KernelHooks,
    HookFilter,
    FrequencyFilter,
    TimeWindowFilter,
    ConditionalFilter,
)

from .registry import (
    ToolRegistry,
    ToolLifecycleManager,
    ToolRegistrationError,
    ToolNotFoundError,
    ToolLifecycleError,
)

# Layer 1 & 2: Tools
from .additional_tools import (
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier,
)

# Public API surface - what users should import
__all__ = [
    # Version
    "__version__",
    
    # Layer 0: Core Infrastructure
    "Tool",
    "ToolConfig",
    "ToolState",
    "ToolMetrics",
    "ToolHandle",
    "AnalysisTool",
    "InterceptorTool",
    "MonitoringTool",
    "SynthesisTool",
    "HookPoint",
    "HookContext",
    "HookTiming",
    "HookPriority",
    "HookRegistry",
    "VMHooks",
    "KernelHooks",
    "HookFilter",
    "FrequencyFilter",
    "TimeWindowFilter",
    "ConditionalFilter",
    "ToolRegistry",
    "ToolLifecycleManager",
    "ToolRegistrationError",
    "ToolNotFoundError",
    "ToolLifecycleError",
    
    # Layer 1 & 2: MVP Tools
    "PatternPrevalenceQuantifier",
    "LocalEntropyMicroscope",
    "EventSignatureClassifier",
]


def get_version() -> str:
    """Get the current version of the void-state-tools package."""
    return __version__


def list_available_tools() -> dict:
    """
    List all available tools organized by layer.
    
    Returns:
        Dictionary mapping layer names to lists of tool classes.
    """
    return {
        "Layer 0: Integration Substrate": [
            "ToolRegistry",
            "HookRegistry",
            "ToolLifecycleManager",
        ],
        "Layer 1 & 2: Analysis Tools": [
            "PatternPrevalenceQuantifier",
            "LocalEntropyMicroscope",
            "EventSignatureClassifier",
        ],
    }


def get_system_info() -> dict:
    """
    Get system information and status.
    
    Returns:
        Dictionary with system version, layer count, and tool count.
    """
    tools = list_available_tools()
    total_tools = sum(len(v) for v in tools.values())
    
    return {
        "version": __version__,
        "layers": len(tools),
        "total_tools": total_tools,
        "status": "mvp-complete",
    }
