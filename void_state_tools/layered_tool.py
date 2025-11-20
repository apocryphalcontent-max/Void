"""
LayeredTool Mixin for enforcing layer and phase declarations.

This module provides a mixin that validates tool layer and phase declarations
at class definition time, ensuring architectural integrity.
"""

from typing import Optional, Dict, Any
import re


class LayerPhaseMismatchError(Exception):
    """Raised when tool layer/phase declaration doesn't match its docstring."""
    pass


class LayeredToolMeta(type):
    """
    Metaclass that enforces layer and phase validation at class definition time.

    Parses the class docstring for Layer and Phase declarations and compares
    them against the class attributes. Raises descriptive errors if mismatched.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip validation for base classes
        if name in ('LayeredTool', 'Tool', 'AnalysisTool', 'MonitoringTool', 'InterceptorTool', 'SynthesisTool'):
            return cls

        # Extract layer and phase from docstring
        docstring = namespace.get('__doc__', '')
        if not docstring:
            # No docstring - skip validation
            return cls

        docstring_layer = mcs._extract_layer_from_docstring(docstring)
        docstring_phase = mcs._extract_phase_from_docstring(docstring)

        # Get declared layer and phase from class attributes or metadata
        declared_layer = namespace.get('_layer')
        declared_phase = namespace.get('_phase')

        # If not set as class attributes, try to get from get_metadata if it exists
        if declared_layer is None or declared_phase is None:
            # Try to instantiate and get metadata (for tools that define it in get_metadata)
            # This is optional - we'll only validate if both are present
            pass

        # Validate if both docstring and declaration are present
        if docstring_layer is not None and declared_layer is not None:
            if docstring_layer != declared_layer:
                raise LayerPhaseMismatchError(
                    f"Layer mismatch in {name}: "
                    f"docstring declares Layer {docstring_layer} "
                    f"but class defines layer {declared_layer}"
                )

        if docstring_phase is not None and declared_phase is not None:
            if docstring_phase != declared_phase:
                raise LayerPhaseMismatchError(
                    f"Phase mismatch in {name}: "
                    f"docstring declares Phase {docstring_phase} "
                    f"but class defines phase {declared_phase}"
                )

        return cls

    @staticmethod
    def _extract_layer_from_docstring(docstring: str) -> Optional[int]:
        """
        Extract layer number from docstring.

        Looks for patterns like:
        - "Layer: 2"
        - "Layer 2"
        - "Layer: 2 (Analysis & Intelligence)"

        Returns:
            Optional[int]: Layer number if found, None otherwise.
        """
        match = re.search(r'Layer:?\s+(\d+)', docstring, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_phase_from_docstring(docstring: str) -> Optional[int]:
        """
        Extract phase number from docstring.

        Looks for patterns like:
        - "Phase: 1"
        - "Phase 1 (MVP)"
        - "Phase: 1 (MVP)"

        Returns:
            Optional[int]: Phase number if found, None otherwise.
        """
        match = re.search(r'Phase:?\s+(\d+)', docstring, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None


class LayeredTool:
    """
    Mixin class for tools that enforces layer and phase declarations.

    Tools that inherit from this mixin must declare their layer and phase
    in both their docstring and as class attributes, ensuring architectural
    consistency.

    Example:
        >>> class MyTool(LayeredTool, Tool):
        ...     \"\"\"
        ...     My Tool
        ...
        ...     Layer: 2 (Analysis & Intelligence)
        ...     Phase: 1 (MVP)
        ...     \"\"\"
        ...     _layer = 2
        ...     _phase = 1
    """

    _layer: Optional[int] = None
    _phase: Optional[int] = None

    def get_layer(self) -> Optional[int]:
        """
        Get the architectural layer this tool belongs to.

        Returns:
            Optional[int]: Layer number (0-4), or None if not set.
        """
        return self._layer

    def get_phase(self) -> Optional[int]:
        """
        Get the deployment phase this tool belongs to.

        Returns:
            Optional[int]: Phase number (1-3), or None if not set.
        """
        return self._phase

    def validate_layer_phase(self) -> bool:
        """
        Validate that layer and phase are properly declared.

        Returns:
            bool: True if valid, False otherwise.

        Raises:
            LayerPhaseMismatchError: If layer or phase is invalid.
        """
        if self._layer is not None:
            if not (0 <= self._layer <= 4):
                raise LayerPhaseMismatchError(
                    f"Invalid layer {self._layer}: must be 0-4"
                )

        if self._phase is not None:
            if not (1 <= self._phase <= 3):
                raise LayerPhaseMismatchError(
                    f"Invalid phase {self._phase}: must be 1-3"
                )

        return True

    def get_architectural_metadata(self) -> Dict[str, Any]:
        """
        Get architectural metadata for this tool.

        Returns:
            dict: Metadata including layer, phase, and descriptions.
        """
        layer_names = {
            0: "Integration Substrate",
            1: "Sensing & Instrumentation",
            2: "Analysis & Intelligence",
            3: "Cognitive & Predictive",
            4: "Meta & Evolution"
        }

        phase_names = {
            1: "MVP",
            2: "Growth",
            3: "Advanced"
        }

        return {
            "layer": self._layer,
            "layer_name": layer_names.get(self._layer, "Unknown"),
            "phase": self._phase,
            "phase_name": phase_names.get(self._phase, "Unknown"),
        }


# Re-export for convenience
__all__ = ['LayeredTool', 'LayeredToolMeta', 'LayerPhaseMismatchError']
