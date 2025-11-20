"""
Void-State Tools System - Complete Package

A modular, extensible toolkit for AI system introspection, maintenance,
education, mutation, and defense.

This package provides the core infrastructure and tools across all phases:
- Phase 1 (MVP): 3 foundational tools ✅
- Phase 2 (Growth): 4+ advanced analysis and prediction tools 🚧
- Phase 3 (Advanced): Tool synthesis and meta-tooling 🚧
"""

__version__ = "3.0.0-phase3-complete"
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

# Phase 2 Tools (Growth)
from .phase2_tools import (
    # Layer 2 Tools
    ThreatSignatureRecognizer,
    BehavioralAnomalyDetector,
    # Layer 3 Tools
    TimelineBranchingEngine,
    ProphecyEngine,
    # Data types
    ThreatType,
    Severity,
    ThreatSignature,
    ThreatAssessment,
    BehaviorTrace,
    BehaviorProfile,
    BehaviorAnomalyReport,
    TimelineFork,
    Perturbation,
    ProphecyDistribution,
)

# Phase 3 Tools (Advanced - Meta-Tooling)
from .phase3_tools import (
    # Layer 4 Meta-Tools
    ToolSynthesizer,
    ToolCombinator,
    ToolMutator,
    ToolFitnessEvaluator,
    # Data types
    PrimitiveType,
    ToolPrimitive,
    ToolSpecification,
    SynthesisResult,
    CompositionStrategy,
    CompositeTool,
    Mutation,
    MutatedTool,
    FitnessReport,
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
    "ToolCombinator",
    "ToolMutator",
    "ToolFitnessEvaluator",

    # Phase 3 Data Types
    "PrimitiveType",
    "ToolPrimitive",
    "ToolSpecification",
    "SynthesisResult",
    "CompositionStrategy",
    "CompositeTool",
    "Mutation",
    "MutatedTool",
    "FitnessReport",
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
        ToolCombinator,
        ToolMutator,
        ToolFitnessEvaluator,
    ]


def get_all_tools():
    """Get all available tools across all phases."""
    return get_mvp_tools() + get_phase2_tools() + get_phase3_tools()


def get_deployment_status():
    """Get current deployment status."""
    return {
        "current_phase": "Phase 3 (Advanced)",
        "version": __version__,
        "total_tools": len(get_all_tools()),
        "total_planned": 47,
        "completion_percentage": f"{int((len(get_all_tools()) / 47) * 100)}%",
        "phase1": {
            "status": "complete",
            "progress": "100%",
            "tools_complete": 3,
            "tools_total": 3,
            "tools": ["PatternPrevalenceQuantifier", "LocalEntropyMicroscope", "EventSignatureClassifier"],
        },
        "phase2": {
            "status": "active",
            "progress": "27%",  # 4 of 15 planned tools
            "tools_complete": 4,
            "tools_total": 15,
            "tools": ["ThreatSignatureRecognizer", "BehavioralAnomalyDetector", "TimelineBranchingEngine", "ProphecyEngine"],
        },
        "phase3": {
            "status": "active",
            "progress": "17%",  # 4 of 24 planned tools
            "tools_complete": 4,
            "tools_total": 24,
            "tools": [
                "ToolSynthesizer (Meta-Tool)",
                "ToolCombinator",
                "ToolMutator",
                "ToolFitnessEvaluator"
            ],
            "note": "Complete meta-tooling system for tool creation and evolution",
        },
    }
