"""
Void-State Tools System - MVP Package

A modular, extensible toolkit for AI system introspection, maintenance,
education, mutation, and defense.

This package provides the core infrastructure and Phase 1 (MVP) tools for
the Void-State system.
"""

__version__ = "1.0.0-mvp-complete"
__author__ = "Void-State Project"
__license__ = "Proprietary"

# Core infrastructure
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

from .registry import (
    ToolRegistry,
    ToolLifecycleManager,
    ToolRegistrationError,
    ToolNotFoundError,
    ToolLifecycleError,
)

from .hooks import (
    HookPoint,
    HookContext,
    HookTiming,
    HookPriority,
    HookRegistry,
)

# MVP Tools (Phase 1)
from .mvp_tools import (
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier,
)

# Public API surface
__all__ = [
    # Version
    "__version__",

    # Core classes
    "Tool",
    "ToolConfig",
    "ToolState",
    "ToolMetrics",
    "ToolHandle",

    # Tool types
    "AnalysisTool",
    "InterceptorTool",
    "MonitoringTool",
    "SynthesisTool",

    # Registry
    "ToolRegistry",
    "ToolLifecycleManager",

    # Exceptions
    "ToolRegistrationError",
    "ToolNotFoundError",
    "ToolLifecycleError",

    # Hooks
    "HookPoint",
    "HookContext",
    "HookTiming",
    "HookPriority",
    "HookRegistry",

    # MVP Tools
    "PatternPrevalenceQuantifier",
    "LocalEntropyMicroscope",
    "EventSignatureClassifier",
]


def get_version() -> str:
    """Get the version string."""
    return __version__


def get_mvp_tools():
    """Get list of available MVP tools."""
    return [
        PatternPrevalenceQuantifier,
        LocalEntropyMicroscope,
        EventSignatureClassifier,
    ]


def get_deployment_status():
    """Get current deployment status."""
    return {
        "current_phase": "Phase 1 (MVP)",
        "version": __version__,
        "phase1": {
            "status": "complete",
            "progress": "100%",
            "tools_complete": 3,
            "tools_total": 3,
        },
        "phase2": {
            "status": "planned",
            "progress": "0%",
        },
        "phase3": {
            "status": "planned",
            "progress": "0%",
        },
    }
