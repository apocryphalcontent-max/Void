"""
Void-State Tools System - Complete Package

A modular, extensible toolkit for AI system introspection, maintenance,
education, mutation, and defense.

This package provides the core infrastructure and tools across all phases:
- Phase 1 (MVP): 3 foundational tools ✅
- Phase 2 (Growth): 4+ advanced analysis and prediction tools 🚧
- Phase 3 (Advanced): Tool synthesis and meta-tooling 🚧
"""

__version__ = "2.0.0-phase2-active"
__author__ = "Void-State Project"
__license__ = "Proprietary"

# Core infrastructure
from .base import (
    AnalysisTool,
    InterceptorTool,
    MonitoringTool,
    SynthesisTool,
    Tool,
    ToolConfig,
    ToolHandle,
    ToolMetrics,
    ToolState,
)
from .hooks import (
    HookContext,
    HookPoint,
    HookPriority,
    HookRegistry,
    HookTiming,
)

# MVP Tools (Phase 1)
from .mvp_tools import (
    EventSignatureClassifier,
    LocalEntropyMicroscope,
    PatternPrevalenceQuantifier,
)

# Phase 2 Tools (Growth)
from .phase2_tools import (
    BehavioralAnomalyDetector,
    BehaviorAnomalyReport,
    BehaviorProfile,
    BehaviorTrace,
    Perturbation,
    ProphecyDistribution,
    ProphecyEngine,
    Severity,
    ThreatAssessment,
    ThreatSignature,
    # Layer 2 Tools
    ThreatSignatureRecognizer,
    # Data types
    ThreatType,
    # Layer 3 Tools
    TimelineBranchingEngine,
    TimelineFork,
)

# Phase 3 Tools (Advanced - Meta-Tooling)
from .phase3_tools import (
    # Data types
    PrimitiveType,
    SynthesisResult,
    ToolPrimitive,
    ToolSpecification,
    # Layer 4 Meta-Tools
    ToolSynthesizer,
)
from .registry import (
    ToolLifecycleError,
    ToolLifecycleManager,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
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

    # Phase 1 MVP Tools
    "PatternPrevalenceQuantifier",
    "LocalEntropyMicroscope",
    "EventSignatureClassifier",

    # Phase 2 Growth Tools
    "ThreatSignatureRecognizer",
    "BehavioralAnomalyDetector",
    "TimelineBranchingEngine",
    "ProphecyEngine",

    # Phase 2 Data Types
    "ThreatType",
    "Severity",
    "ThreatSignature",
    "ThreatAssessment",
    "BehaviorTrace",
    "BehaviorProfile",
    "BehaviorAnomalyReport",
    "TimelineFork",
    "Perturbation",
    "ProphecyDistribution",

    # Phase 3 Advanced Tools
    "ToolSynthesizer",

    # Phase 3 Data Types
    "PrimitiveType",
    "ToolPrimitive",
    "ToolSpecification",
    "SynthesisResult",
]


def get_version() -> str:
    """Get the version string."""
    return __version__


def get_mvp_tools():
    """Get list of available Phase 1 (MVP) tools."""
    return [
        PatternPrevalenceQuantifier,
        LocalEntropyMicroscope,
        EventSignatureClassifier,
    ]


def get_phase2_tools():
    """Get list of available Phase 2 (Growth) tools."""
    return [
        ThreatSignatureRecognizer,
        BehavioralAnomalyDetector,
        TimelineBranchingEngine,
        ProphecyEngine,
    ]


def get_phase3_tools():
    """Get list of available Phase 3 (Advanced) tools."""
    return [
        ToolSynthesizer,
    ]


def get_all_tools():
    """Get all available tools across all phases."""
    return get_mvp_tools() + get_phase2_tools() + get_phase3_tools()


def get_deployment_status():
    """Get current deployment status."""
    # Calculate phase progress dynamically
    phase1_tools = len(get_mvp_tools())
    phase1_total = 3
    phase2_tools = len(get_phase2_tools())
    phase2_total = 15
    phase3_tools = len(get_phase3_tools())
    phase3_total = 24
    total_planned = phase1_total + phase2_total + phase3_total

    return {
        "current_phase": "Phase 3 (Advanced)",
        "version": __version__,
        "total_tools": len(get_all_tools()),
        "total_planned": total_planned,
        "completion_percentage": f"{int((len(get_all_tools()) / total_planned) * 100)}%",
        "phase1": {
            "status": "complete",
            "progress": f"{int((phase1_tools / phase1_total) * 100)}%",
            "tools_complete": phase1_tools,
            "tools_total": phase1_total,
            "tools": ["PatternPrevalenceQuantifier", "LocalEntropyMicroscope", "EventSignatureClassifier"],
        },
        "phase2": {
            "status": "active",
            "progress": f"{int((phase2_tools / phase2_total) * 100)}%",
            "tools_complete": phase2_tools,
            "tools_total": phase2_total,
            "tools": ["ThreatSignatureRecognizer", "BehavioralAnomalyDetector", "TimelineBranchingEngine", "ProphecyEngine"],
        },
        "phase3": {
            "status": "active",
            "progress": f"{int((phase3_tools / phase3_total) * 100)}%",
            "tools_complete": phase3_tools,
            "tools_total": phase3_total,
            "tools": ["ToolSynthesizer (Meta-Tool)"],
            "note": "ToolSynthesizer can generate remaining tools",
        },
    }
