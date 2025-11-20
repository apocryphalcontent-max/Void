"""
Comprehensive tests for Phase 3 (Advanced) tools.

Tests:
- ToolSynthesizer
"""

import pytest
from void_state_tools import (
    ToolConfig,
    ToolSynthesizer,
    PrimitiveType,
    ToolPrimitive,
    ToolSpecification,
)
from void_state_tools.clock import DeterministicClock


class TestToolSynthesizer:
    """Test ToolSynthesizer functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="synthesizer_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolSynthesizer(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 4
        assert tool.get_phase() == 3

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="synthesizer_test")
        tool = ToolSynthesizer(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Tool Synthesizer"
        assert meta["layer"] == 4
        assert meta["phase"] == 3
        assert meta["priority"] == "P0"
        assert "tool_synthesis" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test lifecycle methods."""
        config = ToolConfig(tool_name="synthesizer_test")
        tool = ToolSynthesizer(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_default_primitives_loaded(self):
        """Test default primitives are loaded."""
        config = ToolConfig(tool_name="synthesizer_test")
        tool = ToolSynthesizer(config)

        stats = tool.get_statistics()

        assert stats["registered_primitives"] > 0
        # Should have pattern_match, calculate_mean, classify_threshold, detect_outlier
        assert stats["registered_primitives"] >= 4

    def test_primitive_registration(self):
        """Test registering custom primitives."""
        config = ToolConfig(tool_name="synthesizer_test")
        tool = ToolSynthesizer(config)

        # Register custom primitive
        def custom_transform(x):
            return x * 2

        custom_primitive = ToolPrimitive(
            primitive_id="custom_double",
            primitive_type=PrimitiveType.TRANSFORMATION,
            implementation=custom_transform,
            input_types=[float],
            output_type=float,
            complexity="O(1)",
            description="Double the input value"
        )

        initial_count = tool.get_statistics()["registered_primitives"]
        tool.register_primitive(custom_primitive)

        assert tool.get_statistics()["registered_primitives"] == initial_count + 1

    def test_tool_synthesis_simple(self):
        """Test synthesizing a simple tool."""
        config = ToolConfig(tool_name="synthesizer_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolSynthesizer(config, clock=clock)

        # Create simple tool specification
        spec = ToolSpecification(
            tool_name="SimpleAnalyzer",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Auto-generated simple analyzer",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean"],
            composition_plan=[
                ("calculate_mean", {}),
            ],
            complexity_target="O(N)",
            overhead_target_ms=1.0,
        )

        result = tool.synthesize_tool(spec)

        assert result.success is True
        assert result.tool_class is not None
        assert result.tool_code != ""
        assert len(result.errors) == 0

    def test_tool_synthesis_with_multiple_primitives(self):
        """Test synthesizing tool with multiple primitives."""
        config = ToolConfig(tool_name="synthesizer_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolSynthesizer(config, clock=clock)

        # Create specification with multiple primitives
        spec = ToolSpecification(
            tool_name="ComplexAnalyzer",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Multi-primitive analyzer",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean", "classify_threshold"],
            composition_plan=[
                ("calculate_mean", {}),
                ("classify_threshold", {"threshold": 0.5}),
            ],
            complexity_target="O(N)",
            overhead_target_ms=2.0,
        )

        result = tool.synthesize_tool(spec)

        assert result.success is True
        assert result.tool_class is not None
        assert "calculate_mean" in result.tool_code
        assert "classify_threshold" in result.tool_code

    def test_synthesized_tool_instantiation(self):
        """Test that synthesized tools can be instantiated."""
        config = ToolConfig(tool_name="synthesizer_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolSynthesizer(config, clock=clock)

        spec = ToolSpecification(
            tool_name="TestTool",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Test tool for instantiation",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean"],
            composition_plan=[("calculate_mean", {})],
            complexity_target="O(N)",
            overhead_target_ms=1.0,
        )

        result = tool.synthesize_tool(spec)

        assert result.success is True

        # Try to instantiate the synthesized tool
        tool_config = ToolConfig(tool_name="test_instance")
        instance = result.tool_class(tool_config)

        assert instance is not None
        assert hasattr(instance, "analyze")
        assert hasattr(instance, "get_statistics")

    def test_synthesized_tool_execution(self):
        """Test that synthesized tools can execute analyze()."""
        config = ToolConfig(tool_name="synthesizer_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolSynthesizer(config, clock=clock)

        spec = ToolSpecification(
            tool_name="ExecutableTool",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Test tool execution",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean"],
            composition_plan=[("calculate_mean", {})],
            complexity_target="O(N)",
            overhead_target_ms=1.0,
        )

        result = tool.synthesize_tool(spec)
        assert result.success is True

        # Instantiate and run
        tool_config = ToolConfig(tool_name="test_exec")
        instance = result.tool_class(tool_config)

        test_data = {
            "values": [1.0, 2.0, 3.0, 4.0, 5.0]
        }

        analyze_result = instance.analyze(test_data)

        assert isinstance(analyze_result, dict)
        assert "mean" in analyze_result
        assert analyze_result["mean"] == 3.0

    def test_synthesis_validation(self):
        """Test specification validation."""
        config = ToolConfig(tool_name="synthesizer_test")
        tool = ToolSynthesizer(config)

        # Invalid specification - unknown primitive
        spec = ToolSpecification(
            tool_name="InvalidTool",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Tool with unknown primitive",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["nonexistent_primitive"],
            composition_plan=[("nonexistent_primitive", {})],
            complexity_target="O(N)",
            overhead_target_ms=1.0,
        )

        result = tool.synthesize_tool(spec)

        assert result.success is False
        assert len(result.errors) > 0
        assert "unknown primitive" in result.errors[0].lower()

    def test_synthesis_invalid_layer(self):
        """Test synthesis with invalid layer."""
        config = ToolConfig(tool_name="synthesizer_test")
        tool = ToolSynthesizer(config)

        # Invalid layer
        spec = ToolSpecification(
            tool_name="BadLayerTool",
            tool_type="AnalysisTool",
            layer=99,  # Invalid!
            phase=2,
            description="Tool with bad layer",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean"],
            composition_plan=[("calculate_mean", {})],
            complexity_target="O(N)",
            overhead_target_ms=1.0,
        )

        result = tool.synthesize_tool(spec)

        assert result.success is False
        assert len(result.errors) > 0

    def test_synthesis_statistics(self):
        """Test synthesis statistics tracking."""
        config = ToolConfig(tool_name="synthesizer_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolSynthesizer(config, clock=clock)

        initial_stats = tool.get_statistics()
        assert initial_stats["total_syntheses"] == 0

        # Synthesize a tool
        spec = ToolSpecification(
            tool_name="StatTestTool",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Statistics test",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean"],
            composition_plan=[("calculate_mean", {})],
            complexity_target="O(N)",
            overhead_target_ms=1.0,
        )

        result = tool.synthesize_tool(spec)

        updated_stats = tool.get_statistics()
        assert updated_stats["total_syntheses"] == 1
        assert updated_stats["successful_syntheses"] == (1 if result.success else 0)

    def test_synthesize_method(self):
        """Test synthesize method (SynthesisTool interface)."""
        config = ToolConfig(tool_name="synthesizer_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolSynthesizer(config, clock=clock)

        # Use synthesize() method with dict input
        data = {
            "tool_name": "DictTestTool",
            "tool_type": "AnalysisTool",
            "layer": 2,
            "phase": 2,
            "description": "Test via synthesize method",
            "input_signature": "Dict[str, Any]",
            "output_signature": "Dict[str, Any]",
            "required_primitives": ["calculate_mean"],
            "composition_plan": [("calculate_mean", {})],
            "complexity_target": "O(N)",
            "overhead_target_ms": 1.0,
        }

        result = tool.synthesize(data)

        assert isinstance(result, dict)
        assert "success" in result
        assert "tool_name" in result
        assert result["tool_name"] == "DictTestTool"

    def test_code_generation_quality(self):
        """Test that generated code has proper structure."""
        config = ToolConfig(tool_name="synthesizer_test")
        tool = ToolSynthesizer(config)

        spec = ToolSpecification(
            tool_name="QualityTestTool",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Test code quality",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean"],
            composition_plan=[("calculate_mean", {})],
            complexity_target="O(N)",
            overhead_target_ms=1.0,
        )

        result = tool.synthesize_tool(spec)

        assert result.success is True
        code = result.tool_code

        # Check for essential components
        assert "class QualityTestTool" in code
        assert "def __init__" in code
        assert "def analyze" in code
        assert "LayeredTool" in code
        assert "AnalysisTool" in code
        assert "_layer = 2" in code
        assert "_phase = 2" in code


class TestIntegration:
    """Integration tests for Phase 3 tools."""

    def test_synthesizer_with_registry(self):
        """Test ToolSynthesizer works with ToolRegistry."""
        from void_state_tools import ToolRegistry

        clock = DeterministicClock(start_time=1000.0)
        registry = ToolRegistry(enable_resource_governor=True, clock=clock)

        # Create synthesizer
        config = ToolConfig(tool_name="synthesizer")
        tool = ToolSynthesizer(config, clock=clock)

        # Register
        handle = registry.register_tool(tool)
        registry.lifecycle_manager.attach_tool(handle.tool_id)

        # Verify registered
        assert len(registry._tools) == 1

    def test_recursive_synthesis_concept(self):
        """Test the concept of recursive tool synthesis."""
        config = ToolConfig(tool_name="synthesizer_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolSynthesizer(config, clock=clock)

        # Synthesize first tool
        spec1 = ToolSpecification(
            tool_name="GenTool1",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="First generated tool",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean"],
            composition_plan=[("calculate_mean", {})],
            complexity_target="O(N)",
            overhead_target_ms=1.0,
        )

        result1 = tool.synthesize_tool(spec1)
        assert result1.success is True

        # Synthesize second tool (demonstrating capability to generate multiple)
        spec2 = ToolSpecification(
            tool_name="GenTool2",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Second generated tool",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["classify_threshold"],
            composition_plan=[("classify_threshold", {"threshold": 0.7})],
            complexity_target="O(1)",
            overhead_target_ms=0.5,
        )

        result2 = tool.synthesize_tool(spec2)
        assert result2.success is True

        # Both tools should be different
        assert result1.tool_class != result2.tool_class

        # Statistics should show 2 syntheses
        stats = tool.get_statistics()
        assert stats["total_syntheses"] == 2
