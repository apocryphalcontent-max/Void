# Void-State Tools: Contributing Guide

## Welcome

Thank you for your interest in contributing to the Void-State Proprietary Tools system!

## Development Setup

### Prerequisites
- Python 3.9+
- Git
- pytest (for testing)

### Clone and Install

```bash
git clone https://github.com/apocryphalcontent-max/Messy.git
cd Messy
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest void_state_tools/tests/ -v

# Run with coverage
pytest void_state_tools/tests/ --cov=void_state_tools --cov-report=html

# Run specific test file
pytest void_state_tools/tests/test_mvp.py -v
```

### Run Benchmarks

```bash
python -m void_state_tools.benchmarks
```

## Project Structure

```
void_state_tools/
├── base.py              # Core abstractions
├── registry.py          # Tool registry
├── hooks.py             # Hook system
├── monitoring.py        # Metrics & monitoring
├── mvp/                 # Phase 1 tools
├── phase2/              # Phase 2 tools (planned)
├── phase3/              # Phase 3 tools (planned)
├── tests/               # Test suite
├── benchmarks/          # Performance benchmarks
├── config_examples/     # Configuration templates
└── docs/                # Documentation
```

## Contributing

### 1. Find an Issue

Look for issues labeled:
- `good-first-issue` - Good for newcomers
- `help-wanted` - We need help with this
- `phase-2` - Phase 2 tool implementation
- `enhancement` - New features

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

Follow these guidelines:

#### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to all public functions
- Keep functions focused and small

#### Tool Implementation
When implementing a new tool:

```python
from void_state_tools.base import Tool, ToolConfig
from typing import Dict, Any

class YourTool(Tool):
    """
    Brief description of your tool.
    
    Phase: X (MVP/Growth/Advanced)
    Layer: X (0-4)
    Priority: PX (P0-P3)
    Status: IN_DEVELOPMENT
    """
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        # Initialize your tool
    
    def initialize(self) -> bool:
        """Initialize resources"""
        return True
    
    def shutdown(self) -> bool:
        """Cleanup resources"""
        return True
    
    def suspend(self) -> bool:
        """Suspend operation"""
        return True
    
    def resume(self) -> bool:
        """Resume operation"""
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Your Tool Name",
            "category": "tool_category",
            "version": "1.0.0",
            "description": "What your tool does",
            "capabilities": {"capability1", "capability2"},
            "dependencies": set(),
            "layer": X,
            "phase": X,
            "priority": "PX"
        }
```

#### Testing
Add tests for your code:

```python
# void_state_tools/tests/test_your_tool.py
import pytest
from void_state_tools.base import ToolConfig
from void_state_tools.your_module import YourTool

class TestYourTool:
    def test_initialization(self):
        tool = YourTool(ToolConfig())
        assert tool.initialize()
    
    def test_functionality(self):
        tool = YourTool(ToolConfig())
        tool.initialize()
        # Test your tool's functionality
        assert tool.some_method() == expected_result
```

#### Documentation
- Update README if adding major features
- Add docstrings to all public APIs
- Update CHANGELOG.md
- Add examples if applicable

### 4. Run Quality Checks

```bash
# Run tests
pytest void_state_tools/tests/ -v

# Check test coverage
pytest --cov=void_state_tools --cov-report=term-missing

# Run benchmarks if you changed performance-critical code
python -m void_state_tools.benchmarks
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: Add YourTool implementation

- Implements YourTool for phase X
- Adds tests with 80%+ coverage
- Updates documentation
- Benchmarks show < 100ns overhead"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding tests
- `perf:` - Performance improvements
- `refactor:` - Code refactoring

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Reference to related issues
- Test results
- Benchmark results (if applicable)

## Code Review Process

1. Automated checks run (tests, linting)
2. Maintainer reviews code
3. Address feedback
4. Merge when approved

## Performance Requirements

All tools must meet these overhead budgets:
- **Per-cycle hooks**: < 100ns
- **Per-event hooks**: < 1µs
- **Per-snapshot hooks**: < 10ms
- **Tool memory**: < 100MB default
- **Tool CPU**: < 10% default

Verify with benchmarks:
```bash
python -m void_state_tools.benchmarks
```

## Documentation Standards

### Docstring Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of function.
    
    Longer description if needed, explaining behavior,
    edge cases, and usage examples.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When this exception is raised
    
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output
    """
```

## Phase Roadmap

### Phase 1 (MVP) - COMPLETE ✅
- 6 tools implemented
- Test suite
- Benchmarks
- Monitoring
- Documentation

### Phase 2 (Growth) - IN PROGRESS
Priority tools to implement:
1. Timeline Branching Engine (P1)
2. Prophecy Engine (P1)
3. Threat Signature Recognizer (P0)
4. Behavioral Anomaly Detector (P1)
5. Semantic Memory Diff Analyzer (P1)

### Phase 3 (Advanced) - PLANNED
Meta-tooling focus:
1. Tool Synthesizer (P0)
2. Protocol Synthesis Engine (P1)
3. Tool Combinator (P1)

## Questions?

- Check existing documentation
- Look at example implementations
- Ask in GitHub issues
- Review Phase roadmap in VOID_STATE_STARTUP_ROADMAP.md

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to Void-State Tools!
