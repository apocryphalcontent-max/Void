"""
Comprehensive tests for Phase 3 (Advanced) tools.

Tests:
- ToolSynthesizer
- ToolCombinator
- ToolMutator
- ToolFitnessEvaluator
"""

import pytest
from void_state_tools import (
    ToolConfig,
    ToolSynthesizer,
    ToolCombinator,
    ToolMutator,
    ToolFitnessEvaluator,
    PrimitiveType,
    ToolPrimitive,
    ToolSpecification,
    CompositionStrategy,
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
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


class TestToolCombinator:
    """Test ToolCombinator functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="combinator_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolCombinator(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 4
        assert tool.get_phase() == 3

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="combinator_test")
        tool = ToolCombinator(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Tool Combinator"
        assert meta["layer"] == 4
        assert meta["phase"] == 3
        assert meta["priority"] == "P1"
        assert "tool_composition" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test lifecycle methods."""
        config = ToolConfig(tool_name="combinator_test")
        tool = ToolCombinator(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_register_tool(self):
        """Test registering tools for composition."""
        config = ToolConfig(tool_name="combinator_test")
        combinator = ToolCombinator(config)

        # Register some tools
        tool1_config = ToolConfig(tool_name="tool1")
        tool1 = PatternPrevalenceQuantifier(tool1_config)

        tool2_config = ToolConfig(tool_name="tool2")
        tool2 = LocalEntropyMicroscope(tool2_config)

        combinator.register_tool(tool1)
        combinator.register_tool(tool2)

        stats = combinator.get_statistics()
        assert stats["registered_tools"] == 2

    def test_pipeline_composition(self):
        """Test pipeline composition strategy."""
        config = ToolConfig(tool_name="combinator_test")
        clock = DeterministicClock(start_time=1000.0)
        combinator = ToolCombinator(config, clock=clock)

        # Register tools
        tool1_config = ToolConfig(tool_name="tool1")
        tool1 = PatternPrevalenceQuantifier(tool1_config)

        tool2_config = ToolConfig(tool_name="tool2")
        tool2 = LocalEntropyMicroscope(tool2_config)

        combinator.register_tool(tool1)
        combinator.register_tool(tool2)

        # Create pipeline strategy
        strategy = CompositionStrategy(
            strategy_type="pipeline",
            dataflow_graph={}
        )

        # Combine tools
        composite = combinator.combine_tools(["tool1", "tool2"], strategy)

        assert composite.name.startswith("Composite_pipeline")
        assert len(composite.component_tools) == 2
        assert "tool1" in composite.component_tools
        assert "tool2" in composite.component_tools
        assert composite.composition_graph["tool1"] == ["tool2"]

    def test_parallel_composition(self):
        """Test parallel composition strategy."""
        config = ToolConfig(tool_name="combinator_test")
        clock = DeterministicClock(start_time=1000.0)
        combinator = ToolCombinator(config, clock=clock)

        # Register tools
        tool1_config = ToolConfig(tool_name="tool1")
        tool1 = PatternPrevalenceQuantifier(tool1_config)

        tool2_config = ToolConfig(tool_name="tool2")
        tool2 = LocalEntropyMicroscope(tool2_config)

        combinator.register_tool(tool1)
        combinator.register_tool(tool2)

        # Create parallel strategy
        strategy = CompositionStrategy(
            strategy_type="parallel",
            dataflow_graph={}
        )

        # Combine tools
        composite = combinator.combine_tools(["tool1", "tool2"], strategy)

        assert composite.name.startswith("Composite_parallel")
        assert composite.performance_profile["parallelism"] == 2

    def test_synthesize_method(self):
        """Test synthesize method interface."""
        config = ToolConfig(tool_name="combinator_test")
        combinator = ToolCombinator(config)

        # Register tools
        tool1_config = ToolConfig(tool_name="tool1")
        tool1 = PatternPrevalenceQuantifier(tool1_config)
        combinator.register_tool(tool1)

        tool2_config = ToolConfig(tool_name="tool2")
        tool2 = LocalEntropyMicroscope(tool2_config)
        combinator.register_tool(tool2)

        # Call synthesize
        result = combinator.synthesize({
            "tool_names": ["tool1", "tool2"],
            "strategy_type": "pipeline"
        })

        assert "name" in result
        assert result["component_count"] == 2

    def test_invalid_tool_name(self):
        """Test error handling for unregistered tools."""
        config = ToolConfig(tool_name="combinator_test")
        combinator = ToolCombinator(config)

        strategy = CompositionStrategy(
            strategy_type="pipeline",
            dataflow_graph={}
        )

        with pytest.raises(ValueError, match="Tool not registered"):
            combinator.combine_tools(["nonexistent"], strategy)


class TestToolMutator:
    """Test ToolMutator functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="mutator_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolMutator(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 4
        assert tool.get_phase() == 3

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="mutator_test")
        tool = ToolMutator(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Tool Mutator"
        assert meta["layer"] == 4
        assert meta["phase"] == 3
        assert meta["priority"] == "P1"
        assert "tool_mutation" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test lifecycle methods."""
        config = ToolConfig(tool_name="mutator_test")
        tool = ToolMutator(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_register_tool(self):
        """Test registering tools for mutation."""
        config = ToolConfig(tool_name="mutator_test")
        mutator = ToolMutator(config)

        # Register a tool
        tool_config = ToolConfig(tool_name="test_tool")
        tool = PatternPrevalenceQuantifier(tool_config)

        mutator.register_tool(tool)

        stats = mutator.get_statistics()
        assert stats["registered_tools"] == 1

    def test_mutate_tool(self):
        """Test tool mutation."""
        config = ToolConfig(tool_name="mutator_test")
        clock = DeterministicClock(start_time=1000.0)
        mutator = ToolMutator(config, clock=clock)

        # Register a tool
        tool_config = ToolConfig(tool_name="test_tool")
        tool = PatternPrevalenceQuantifier(tool_config)
        mutator.register_tool(tool)

        # Mutate the tool
        mutated = mutator.mutate_tool("test_tool", mutation_budget=3)

        assert mutated.original_tool == "test_tool"
        assert len(mutated.mutations) > 0
        assert mutated.fitness_delta >= 0.0

    def test_mutation_types(self):
        """Test different mutation types."""
        config = ToolConfig(tool_name="mutator_test")
        mutator = ToolMutator(config)

        # Register a tool
        tool_config = ToolConfig(tool_name="test_tool")
        tool = LocalEntropyMicroscope(tool_config)
        mutator.register_tool(tool)

        # Test parameter mutations
        mutated = mutator.mutate_tool(
            "test_tool",
            mutation_budget=2,
            mutation_types=["parameter"]
        )

        assert all(m.mutation_type == "parameter" for m in mutated.mutations)

        # Test optimization mutations
        mutated2 = mutator.mutate_tool(
            "test_tool",
            mutation_budget=2,
            mutation_types=["optimization"]
        )

        assert all(m.mutation_type == "optimization" for m in mutated2.mutations)

    def test_synthesize_method(self):
        """Test synthesize method interface."""
        config = ToolConfig(tool_name="mutator_test")
        mutator = ToolMutator(config)

        # Register a tool
        tool_config = ToolConfig(tool_name="test_tool")
        tool = PatternPrevalenceQuantifier(tool_config)
        mutator.register_tool(tool)

        # Call synthesize
        result = mutator.synthesize({
            "tool_name": "test_tool",
            "mutation_budget": 3
        })

        assert result["original_tool"] == "test_tool"
        assert result["mutations_applied"] > 0

    def test_invalid_tool_name(self):
        """Test error handling for unregistered tools."""
        config = ToolConfig(tool_name="mutator_test")
        mutator = ToolMutator(config)

        with pytest.raises(ValueError, match="Tool not registered"):
            mutator.mutate_tool("nonexistent")


class TestToolFitnessEvaluator:
    """Test ToolFitnessEvaluator functionality."""

    def test_instantiation(self):
        """Test tool can be instantiated."""
        config = ToolConfig(tool_name="evaluator_test")
        clock = DeterministicClock(start_time=1000.0)
        tool = ToolFitnessEvaluator(config, clock=clock)

        assert tool is not None
        assert tool.get_layer() == 4
        assert tool.get_phase() == 3

    def test_metadata(self):
        """Test tool metadata."""
        config = ToolConfig(tool_name="evaluator_test")
        tool = ToolFitnessEvaluator(config)
        meta = tool.get_metadata()

        assert meta["name"] == "Tool Fitness Evaluator"
        assert meta["layer"] == 4
        assert meta["phase"] == 3
        assert meta["priority"] == "P1"
        assert "fitness_evaluation" in meta["capabilities"]

    def test_lifecycle_methods(self):
        """Test lifecycle methods."""
        config = ToolConfig(tool_name="evaluator_test")
        tool = ToolFitnessEvaluator(config)

        assert tool.initialize() is True
        assert tool.suspend() is True
        assert tool.resume() is True
        assert tool.shutdown() is True

    def test_register_tool(self):
        """Test registering tools for evaluation."""
        config = ToolConfig(tool_name="evaluator_test")
        evaluator = ToolFitnessEvaluator(config)

        # Register a tool
        tool_config = ToolConfig(tool_name="test_tool")
        tool = PatternPrevalenceQuantifier(tool_config)

        evaluator.register_tool(tool)

        stats = evaluator.get_statistics()
        assert stats["registered_tools"] == 1

    def test_evaluate_fitness(self):
        """Test fitness evaluation."""
        config = ToolConfig(tool_name="evaluator_test")
        clock = DeterministicClock(start_time=1000.0)
        evaluator = ToolFitnessEvaluator(config, clock=clock)

        # Register a tool
        tool_config = ToolConfig(tool_name="test_tool")
        tool = PatternPrevalenceQuantifier(tool_config)
        evaluator.register_tool(tool)

        # Evaluate fitness
        report = evaluator.evaluate_fitness("test_tool")

        assert report.tool_name == "test_tool"
        assert 0.0 <= report.overall_fitness <= 1.0
        assert "performance" in report.metric_scores
        assert "correctness" in report.metric_scores
        assert "robustness" in report.metric_scores
        assert "maintainability" in report.metric_scores
        assert isinstance(report.failure_modes, list)
        assert isinstance(report.recommendations, list)

    def test_evaluate_with_test_data(self):
        """Test fitness evaluation with test data."""
        config = ToolConfig(tool_name="evaluator_test")
        evaluator = ToolFitnessEvaluator(config)

        # Register a tool
        tool_config = ToolConfig(tool_name="test_tool")
        tool = PatternPrevalenceQuantifier(tool_config)
        evaluator.register_tool(tool)

        # Prepare test data
        test_data = [
            {"pattern": "test", "context": "test_context"},
            {"pattern": "test2", "context": "test_context2"}
        ]

        # Evaluate with test data
        report = evaluator.evaluate_fitness("test_tool", test_data)

        assert report.overall_fitness > 0.0

    def test_analyze_method(self):
        """Test analyze method interface."""
        config = ToolConfig(tool_name="evaluator_test")
        evaluator = ToolFitnessEvaluator(config)

        # Register a tool
        tool_config = ToolConfig(tool_name="test_tool")
        tool = LocalEntropyMicroscope(tool_config)
        evaluator.register_tool(tool)

        # Call analyze
        result = evaluator.analyze({
            "tool_name": "test_tool"
        })

        assert result["tool_name"] == "test_tool"
        assert "overall_fitness" in result
        assert "metric_scores" in result
        assert "failure_count" in result

    def test_invalid_tool_name(self):
        """Test error handling for unregistered tools."""
        config = ToolConfig(tool_name="evaluator_test")
        evaluator = ToolFitnessEvaluator(config)

        with pytest.raises(ValueError, match="Tool not registered"):
            evaluator.evaluate_fitness("nonexistent")

    def test_recommendations_generation(self):
        """Test that recommendations are generated."""
        config = ToolConfig(tool_name="evaluator_test")
        evaluator = ToolFitnessEvaluator(config)

        # Register a well-formed tool
        tool_config = ToolConfig(tool_name="good_tool")
        tool = PatternPrevalenceQuantifier(tool_config)
        evaluator.register_tool(tool)

        report = evaluator.evaluate_fitness("good_tool")

        # Should have recommendations
        assert len(report.recommendations) > 0

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        config = ToolConfig(tool_name="evaluator_test")
        evaluator = ToolFitnessEvaluator(config)

        # Register tools
        tool1_config = ToolConfig(tool_name="tool1")
        tool1 = PatternPrevalenceQuantifier(tool1_config)
        evaluator.register_tool(tool1)

        tool2_config = ToolConfig(tool_name="tool2")
        tool2 = LocalEntropyMicroscope(tool2_config)
        evaluator.register_tool(tool2)

        # Evaluate both
        evaluator.evaluate_fitness("tool1")
        evaluator.evaluate_fitness("tool2")

        stats = evaluator.get_statistics()
        assert stats["total_evaluations"] == 2
        assert stats["average_fitness"] > 0.0


class TestPhase3Integration:
    """Integration tests for Phase 3 tools working together."""

    def test_full_meta_tooling_pipeline(self):
        """Test complete meta-tooling pipeline."""
        clock = DeterministicClock(start_time=1000.0)

        # Create synthesizer
        synth_config = ToolConfig(tool_name="synthesizer")
        synthesizer = ToolSynthesizer(synth_config, clock=clock)

        # Create evaluator
        eval_config = ToolConfig(tool_name="evaluator")
        evaluator = ToolFitnessEvaluator(eval_config, clock=clock)

        # Create mutator
        mut_config = ToolConfig(tool_name="mutator")
        mutator = ToolMutator(mut_config, clock=clock)

        # Synthesize a tool
        spec = ToolSpecification(
            tool_name="SynthesizedTool",
            tool_type="AnalysisTool",
            layer=2,
            phase=2,
            description="Test synthesized tool",
            input_signature="Dict[str, Any]",
            output_signature="Dict[str, Any]",
            required_primitives=["calculate_mean"],
            composition_plan=[("calculate_mean", {})],
            complexity_target="O(N)",
            overhead_target_ms=1.0
        )

        result = synthesizer.synthesize_tool(spec)
        assert result.success is True

        # Instantiate the synthesized tool
        tool_config = ToolConfig(tool_name="synth_instance")
        synth_tool = result.tool_class(tool_config, clock=clock)

        # Evaluate the synthesized tool
        evaluator.register_tool(synth_tool)
        fitness_report = evaluator.evaluate_fitness("synth_instance")

        assert fitness_report.overall_fitness > 0.0

        # Mutate the synthesized tool
        mutator.register_tool(synth_tool)
        mutated = mutator.mutate_tool("synth_instance", mutation_budget=2)

        assert len(mutated.mutations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
