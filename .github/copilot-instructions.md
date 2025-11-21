# GitHub Copilot Instructions for Void

This repository contains the **Void: Self-Aware AI System Architecture** - a layered, capability-based framework for AI system introspection, maintenance, education, mutation, and defense.

## Project Overview

Void is a sophisticated Python framework with **47 specialized tools** organized across **5 layers** (0-4), deployed in **3 progressive phases** (MVP, Growth, Advanced). The system implements Byzantine fault tolerance, distributed consensus, advanced type systems, and strict performance overhead budgets.

**Current Status:** MVP Complete (Phase 1) with 3 core tools operational.

## Architecture Principles

When contributing code, always adhere to these core principles:

1. **Layered Composability**: Each layer builds on primitives below
   - Layer 0: Integration Substrate (VM/Kernel hooks, registry, lifecycle)
   - Layer 1: Sensing & Instrumentation (Memory diffing, execution tracing)
   - Layer 2: Analysis & Intelligence (Pattern recognition, anomaly detection)
   - Layer 3: Cognitive & Predictive (Timeline branching, prophecy, threat modeling)
   - Layer 4: Meta & Evolution (Tool synthesis, mutation, self-modification)

2. **Resource-Governed**: Every tool operates within strict CPU/memory/hook quotas
3. **Capability-Based Security**: Tools request minimal privileges via macaroon tokens
4. **Overhead-Budgeted Hooks**: Nanosecond-precision timing enforcement prevents observer effect
5. **Phased Deployment**: Respect the phase boundaries (MVP → Growth → Advanced)

## Code Style & Standards

### Python Standards
- **Python Version**: 3.9+ (support 3.9, 3.10, 3.11, 3.12)
- **Style Guide**: PEP 8 with 120 character line length
- **Formatters**: Use `black` (line-length=120) and `ruff` for linting
- **Type Hints**: All functions must have complete type hints
- **Docstrings**: Required for all public APIs using Google-style format

### Type System Requirements
The project uses advanced type systems:
- **Linear Types**: Resources that must be consumed exactly once (`LinearResource`)
- **Dependent Types**: Types with value-level constraints (`Vector[N]`, `Range[min,max]`, `NonEmpty[T]`)
- **Type Validation**: Metaclass validation at class definition time

### Example Code Pattern
```python
from void_state_tools.base import Tool, ToolConfig
from typing import Dict, Any, Optional

class YourTool(Tool):
    """
    Brief description of your tool.
    
    This tool implements [specific functionality] as part of [Layer X].
    
    Phase: X (MVP/Growth/Advanced)
    Layer: X (0-4)
    Priority: PX (P0-P3)
    Status: IN_DEVELOPMENT
    
    Args:
        config: Tool configuration with resource quotas
        
    Example:
        >>> config = ToolConfig(max_memory_mb=100, max_cpu_percent=10)
        >>> tool = YourTool(config)
        >>> tool.initialize()
        >>> result = tool.process_data(...)
    """
    
    def __init__(self, config: ToolConfig) -> None:
        super().__init__(config)
        # Initialize with type hints
    
    def initialize(self) -> bool:
        """Initialize resources and validate configuration."""
        return True
    
    def shutdown(self) -> bool:
        """Cleanup resources and release capabilities."""
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata including layer, phase, and capabilities."""
        return {
            "name": "Your Tool Name",
            "category": "tool_category",
            "version": "1.0.0",
            "description": "What your tool does",
            "capabilities": {"capability1", "capability2"},
            "dependencies": set(),
            "layer": 0,
            "phase": 1,
            "priority": "P0"
        }
```

## Performance Requirements

**CRITICAL**: All tools must meet strict overhead budgets to prevent observer effect:

| Hook Type | Frequency | Budget | Enforcement |
|-----------|-----------|--------|-------------|
| Per-cycle | ~1 GHz | **100ns** | Detach after 3 violations |
| Per-instruction | ~1 GHz | **50ns** | Statistical sampling |
| Per-event | Variable | **1µs** | Immediate detach |
| Per-snapshot | ~1/min | **10ms** | Throttle + backpressure |

**Default Resource Limits:**
- Tool memory: < 100MB
- Tool CPU: < 10%
- Hook callback overhead: Must not exceed budget 3x consecutively

**Validation:**
```bash
# Always run benchmarks for performance-critical changes
pytest void_state_tools/tests/ --benchmark-only
```

## Testing Requirements

### Test Structure
- **Location**: `void_state_tools/tests/` or `tests/`
- **Framework**: pytest with markers (unit, integration, slow, benchmark)
- **Coverage**: Aim for 80%+ coverage on new code
- **Test Files**: Follow `test_*.py` naming convention

### Test Pattern
```python
import pytest
from void_state_tools.base import ToolConfig
from void_state_tools.your_module import YourTool

class TestYourTool:
    """Test suite for YourTool."""
    
    @pytest.fixture
    def tool_config(self) -> ToolConfig:
        """Create test configuration."""
        return ToolConfig(
            tool_name="test_tool",
            max_memory_mb=50,
            max_cpu_percent=5,
            overhead_budget_ns=500
        )
    
    @pytest.fixture
    def tool(self, tool_config: ToolConfig) -> YourTool:
        """Create tool instance."""
        return YourTool(tool_config)
    
    def test_initialization(self, tool: YourTool) -> None:
        """Test tool initializes correctly."""
        assert tool.initialize()
        assert tool.state == ToolState.ACTIVE
    
    @pytest.mark.benchmark
    def test_performance(self, tool: YourTool, benchmark) -> None:
        """Test tool meets performance requirements."""
        tool.initialize()
        result = benchmark(tool.process_data, sample_data)
        assert result is not None
```

### Running Tests
```bash
# Run all tests
pytest void_state_tools/tests/ -v

# Run with coverage
pytest void_state_tools/tests/ --cov=void_state_tools --cov-report=term

# Run specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
pytest -m benchmark --benchmark-only  # Only benchmarks

# Run validation suite
python validate_hardening.py
```

## Tool Implementation Guidelines

### Tool Lifecycle States
All tools must implement the state machine:
```
DORMANT → INITIALIZING → ACTIVE → SUSPENDED → TERMINATED
```

### Required Methods
- `initialize() -> bool`: Setup and resource allocation
- `shutdown() -> bool`: Cleanup and resource release
- `suspend() -> bool`: Pause operation (maintain state)
- `resume() -> bool`: Resume from suspension
- `get_metadata() -> Dict[str, Any]`: Return tool metadata

### Hook Integration
When implementing hooks:
```python
from void_state_tools.hooks import HookType, HookPriority

def register_hooks(self, hook_manager) -> None:
    """Register tool hooks with overhead budgets."""
    hook_manager.register(
        hook_type=HookType.BEFORE_CYCLE,
        callback=self._on_cycle,
        priority=HookPriority.NORMAL,
        overhead_budget_ns=100  # Must meet budget!
    )
```

### Capability Requests
Use capability-based security:
```python
from void_state_tools.capabilities import CapabilityToken

def request_capability(self) -> CapabilityToken:
    """Request minimal necessary capabilities."""
    return self.capability_manager.create_token(
        permissions=["read_memory", "observe_events"],
        caveats={"max_duration_s": 3600, "rate_limit": 1000}
    )
```

## Distributed Systems

The project uses:
- **Hybrid Logical Clocks (HLC)**: For causal ordering and total order
- **Plumtree Gossip**: O(n) message complexity, eager push/lazy pull
- **PBFT Consensus**: Tolerates f < n/3 Byzantine faults
- **Content-Addressable Storage**: For state immutability

When working with distributed components:
```python
from void_state_tools.distributed import HybridLogicalClock, ConsensusProtocol

# Use HLC for event ordering
hlc = HybridLogicalClock(node_id="node1")
timestamp = hlc.send_event()

# Ensure consensus for critical operations
consensus = ConsensusProtocol(replicas=4, tolerance=1)
result = await consensus.propose(operation)
```

## Security Considerations

### Sandboxing
- All tools run in isolated memory/network namespaces
- Use `seccomp` filters for syscall restrictions
- Circuit breakers auto-suspend on repeated violations

### Cryptographic Signing
- State snapshots must be signed with Ed25519
- Verify signatures before state restoration
- Maintain audit trail for capability usage

### Input Validation
```python
from void_state_tools.advanced_types import NonEmpty, Range

def process_data(
    self,
    items: NonEmpty[List[int]],  # Cannot be empty
    threshold: Range[0, 100]      # Must be 0-100
) -> Result:
    """Process with validated inputs."""
    # Type system enforces constraints
    pass
```

## Documentation Standards

### Module Docstrings
```python
"""
Module description.

This module implements [functionality] for [purpose].
Part of Layer X, Phase Y.

Key Components:
    - ComponentA: Does X
    - ComponentB: Does Y

Example:
    >>> from void_state_tools.module import Component
    >>> component = Component()
    >>> result = component.process()
"""
```

### Function Docstrings
Use Google-style format:
```python
def complex_function(
    param1: Type1,
    param2: Type2,
    optional_param: Optional[Type3] = None
) -> ReturnType:
    """
    Brief description (one line).
    
    Longer description explaining behavior, edge cases,
    and important implementation details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        optional_param: Optional description (default: None)
    
    Returns:
        Description of return value including structure
    
    Raises:
        ValueError: When input validation fails
        ResourceExhaustedError: When quota is exceeded
    
    Example:
        >>> result = complex_function(val1, val2)
        >>> print(result.status)
        'success'
    
    Note:
        Performance: O(n log n) time, O(n) space
        Thread-safe: Yes
        Overhead: < 100ns per call
    """
```

## Common Pitfalls to Avoid

1. **Don't exceed overhead budgets** - Always benchmark performance-critical code
2. **Don't bypass capability system** - Never use direct access when capabilities exist
3. **Don't ignore resource quotas** - Respect memory and CPU limits
4. **Don't break phase boundaries** - Phase 1 code shouldn't depend on Phase 2 features
5. **Don't skip type hints** - The type system is crucial for safety
6. **Don't forget state machine** - All tools must implement lifecycle methods
7. **Don't omit tests** - Coverage and benchmarks are required
8. **Don't modify working tests** - Unless fixing bugs in your changed code

## CI/CD Pipeline

The repository has comprehensive CI/CD:
- **Linting**: ruff + mypy on every push
- **Testing**: pytest across Python 3.9-3.12, Ubuntu/macOS/Windows
- **Benchmarks**: Automated performance validation
- **Security**: bandit + safety scans
- **Coverage**: Codecov integration
- **Documentation**: Markdown link checking

All checks must pass before merge.

## File Organization

```
void_state_tools/
├── base.py              # Core abstractions (Tool, ToolConfig, ToolState)
├── registry.py          # Tool registry and lifecycle manager
├── hooks.py             # Hook system (16 integration points)
├── monitoring.py        # Metrics and observability
├── capabilities.py      # Capability-based security
├── distributed.py       # HLC, Plumtree, PBFT
├── linear_types.py      # Linear type system
├── dependent_types.py   # Dependent type system
├── advanced_types.py    # Type system extensions
├── effects.py           # Effect system
├── tests/               # Test suite
│   ├── test_integration.py
│   ├── test_phase2_tools.py
│   └── test_phase3_tools.py
└── [tool_modules].py    # Individual tool implementations
```

## Dependencies

**Core** (always available):
- numpy >= 2.0.0
- scipy >= 1.10.0
- networkx >= 2.8.0

**Optional** (install as needed):
- `[ml]`: scikit-learn, torch
- `[distributed]`: redis, etcd3
- `[quantum]`: dimod, dwave-ocean-sdk
- `[monitoring]`: prometheus-client, opentelemetry, psutil
- `[test]`: pytest, pytest-cov, pytest-asyncio, pytest-benchmark, hypothesis
- `[dev]`: mypy, black, ruff, pylint

## Phase Roadmap

### Phase 1 (MVP) - COMPLETE ✅
- 3 core tools operational
- Infrastructure complete
- Full test coverage

### Phase 2 (Growth) - IN PROGRESS
Priority tools (P0-P1):
- Timeline Branching Engine
- Prophecy Engine
- Threat Signature Recognizer
- Behavioral Anomaly Detector
- Semantic Memory Diff Analyzer

### Phase 3 (Advanced) - PLANNED
Meta-tooling focus:
- Tool Synthesizer
- Protocol Synthesis Engine
- Recursive Meta-Tool

## Commit Message Format

Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding/modifying tests
- `perf:` - Performance improvements
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks
- `security:` - Security improvements

Example:
```
feat: Implement Timeline Branching Engine

- Add TimelineBranch dataclass with causal links
- Implement branch creation and merging
- Add HLC integration for temporal ordering
- Include tests with 85% coverage
- Benchmarks show 95ns overhead (under 100ns budget)

Closes #42
```

## Getting Help

- **Architecture**: See IMPLEMENTATION_SUMMARY.md, ARCHITECTURE_HARDENING.md
- **Tools**: See VOID_STATE_TOOLS_TAXONOMY.md, VOID_STATE_TOOLS_SPECIFICATION.md
- **API**: See API.md
- **Contributing**: See VOID_STATE_CONTRIBUTING.md
- **Deployment**: See VOID_STATE_DEPLOYMENT_GUIDE.md
- **FAQ**: See VOID_STATE_FAQ.md

## Quick Reference Commands

```bash
# Development setup
pip install -e ".[dev,test]"

# Run linters
ruff check .
mypy void_state_tools --ignore-missing-imports
black --check .

# Run tests
pytest void_state_tools/tests/ -v
pytest --cov=void_state_tools --cov-report=term

# Run benchmarks
pytest --benchmark-only

# Validate hardening
python validate_hardening.py

# Build package
python -m build
```

---

**Remember**: This is a production-grade system with strict requirements. Always prioritize correctness, performance, and security over convenience. When in doubt, consult the extensive documentation in the repository.
