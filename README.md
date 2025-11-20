# Void: Self-Aware AI System Architecture

> **A layered, capability-based framework for AI system introspection, maintenance, education, mutation, and defense.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Phase%203%20Active-brightgreen.svg)]()

## Vision

Just as biological organisms require nervous systems for sensing and immune systems for protection, AI systems need introspective tooling for self-awareness, adaptive maintenance, and autonomous evolution. Void provides this substrate through a modular architecture of **47 specialized tools** organized across **5 layers**, deployed in **3 progressive phases**.

## Architecture at a Glance

```
Layer 4: Meta & Evolution        → Tool synthesis, mutation, self-modification
Layer 3: Cognitive & Predictive  → Timeline branching, prophecy, threat modeling
Layer 2: Analysis & Intelligence → Pattern recognition, anomaly detection
Layer 1: Sensing & Instrumentation → Memory diffing, execution tracing
Layer 0: Integration Substrate   → VM/Kernel hooks, registry, lifecycle
```

**Core Principles:**
- **Layered Composability**: Each layer builds on the primitives below
- **Resource-Governed**: Every tool operates within strict CPU/memory/hook quotas
- **Capability-Based Security**: Tools request minimal privileges via macaroon tokens
- **Overhead-Budgeted Hooks**: Nanosecond-precision timing enforcement prevents observer effect
- **Phased Deployment**: MVP (3 tools) → Growth (+15) → Advanced (+24)

## Current Status: Phase 3 Active (28% Complete)

**Version:** 3.0.0-phase3-complete
**Progress:** 13 of 47 tools implemented

### Phase 1 Tools (3/3) - COMPLETE ✅
- ✅ **PatternPrevalenceQuantifier**: Tracks pattern frequency and ubiquity across system state
- ✅ **LocalEntropyMicroscope**: Measures Shannon entropy at microscopic scales with gradient analysis
- ✅ **EventSignatureClassifier**: Naive Bayes classifier for event taxonomic categorization

### Phase 2 Tools (6/15) - ACTIVE 🚧
- ✅ **ThreatSignatureRecognizer** (Layer 2): Real-time pattern matching against threat signature database
- ✅ **BehavioralAnomalyDetector** (Layer 2): Behavior sequence learning and deviation detection
- ✅ **NoveltyDetector** (Layer 2): Identifies unprecedented patterns through similarity analysis
- ✅ **TimelineBranchingEngine** (Layer 3): State forking and parallel timeline execution
- ✅ **ProphecyEngine** (Layer 3): Forward dynamics simulation with Monte Carlo sampling
- ✅ **ExternalInterferenceDetector** (Layer 3): Detects unauthorized external influences on system state

### Phase 3 Tools (4/24) - ACTIVE 🔬
- ✅ **ToolSynthesizer** (Layer 4): Meta-tool for generating new tools from specifications
- ✅ **ToolCombinator** (Layer 4): Composes multiple tools into pipelines and parallel workflows
- ✅ **ToolMutator** (Layer 4): Evolves tools through controlled mutations and fitness-guided optimization
- ✅ **ToolFitnessEvaluator** (Layer 4): Multi-dimensional assessment of tool quality and performance

**Infrastructure (Complete):**
- Tool Registry & Lifecycle Manager (DORMANT → ACTIVE → SUSPENDED → TERMINATED)
- VM/Kernel Hook System (16 hook points with overhead budgets)
- Linear & Dependent Type System (resource safety, dimensionality checking)
- Distributed Consensus (HLC, Plumtree gossip, PBFT f<n/3 fault tolerance)
- Capability System (chained macaroon tokens with caveats)

## Quick Start

```python
# Install in editable mode
pip install -e .

# Import tools from all phases
from void_state_tools import (
    ToolRegistry, ToolConfig,
    # Phase 1 (MVP)
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier,
    # Phase 2 (Growth)
    ThreatSignatureRecognizer,
    BehavioralAnomalyDetector,
    TimelineBranchingEngine,
    ProphecyEngine,
    # Phase 3 (Advanced Meta-Tools)
    ToolSynthesizer,
    ToolCombinator,
    ToolMutator,
    ToolFitnessEvaluator,
)

# Create registry and register a tool
registry = ToolRegistry()
config = ToolConfig(
    tool_name="pattern_analyzer",
    max_memory_mb=100,
    max_cpu_percent=10,
    overhead_budget_ns=1000
)

# Example: Phase 1 Pattern Analysis
tool = PatternPrevalenceQuantifier(config)
handle = registry.register_tool(tool)
registry.lifecycle_manager.attach_tool(handle.tool_id)

result = tool.analyze({
    "pattern": "memory_spike",
    "context": "inference_loop",
    "timestamp": time.time()
})
print(f"Frequency: {result['frequency_ratio']:.2%}")

# Example: Phase 3 Tool Synthesis
from void_state_tools import ToolSpecification

synthesizer = ToolSynthesizer(ToolConfig(tool_name="synth"))
spec = ToolSpecification(
    tool_name="CustomAnalyzer",
    tool_type="AnalysisTool",
    layer=2, phase=2,
    description="Auto-generated analyzer",
    required_primitives=["pattern_match"],
    composition_plan=[("pattern_match", {})],
)
result = synthesizer.synthesize_tool(spec)
# result.tool_class is a working Tool subclass!
```

## Documentation Map

### Core Architecture
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** — Complete system overview (19,598 LOC across 39 modules)
- **[ARCHITECTURE_HARDENING.md](ARCHITECTURE_HARDENING.md)** — Security, sandboxing, circuit breakers
- **[API.md](API.md)** — Full API reference

### Void-State Tools System
- **[VOID_STATE_TOOLS_TAXONOMY.md](VOID_STATE_TOOLS_TAXONOMY.md)** — Layered organization, implementation status matrix
- **[VOID_STATE_TOOLS_SPECIFICATION.md](VOID_STATE_TOOLS_SPECIFICATION.md)** — 50+ data types, I/O signatures, complexity bounds
- **[VOID_STATE_TOOLS_README.md](VOID_STATE_TOOLS_README.md)** — Tool catalog, integration examples
- **[VOID_STATE_INTEGRATION_ARCHITECTURE.md](VOID_STATE_INTEGRATION_ARCHITECTURE.md)** — VM/Kernel hooks, lifecycle, quotas

### Distributed Systems
- **[DISTRIBUTED_SYSTEM_IMPLEMENTATION.md](DISTRIBUTED_SYSTEM_IMPLEMENTATION.md)** — HLC, Plumtree (O(n) broadcast), PBFT
- **[VOID_STATE_MATHEMATICAL_FOUNDATIONS.md](VOID_STATE_MATHEMATICAL_FOUNDATIONS.md)** — Category theory, causality, entropy

### Deployment & Roadmap
- **[VOID_STATE_STARTUP_ROADMAP.md](VOID_STATE_STARTUP_ROADMAP.md)** — 3-phase deployment (6→18→36 months)
- **[VOID_STATE_QUICKSTART.md](VOID_STATE_QUICKSTART.md)** — Installation, first tool, hook integration
- **[VOID_STATE_DEPLOYMENT_GUIDE.md](VOID_STATE_DEPLOYMENT_GUIDE.md)** — Production deployment patterns
- **[VOID_STATE_FAQ.md](VOID_STATE_FAQ.md)** — Frequently asked questions

### Release Notes & Contributing
- **[VOID_STATE_V2_RELEASE_NOTES.md](VOID_STATE_V2_RELEASE_NOTES.md)** through **[V3.3](VOID_STATE_V3.3_RELEASE_NOTES.md)** — Version history
- **[VOID_STATE_CONTRIBUTING.md](VOID_STATE_CONTRIBUTING.md)** — Contribution guidelines

## Key Features

### 1. Tool Registry & Lifecycle
- **State Machine**: DORMANT → INITIALIZING → ACTIVE → SUSPENDED → TERMINATED
- **Resource Monitoring**: Real-time RSS, CPU via psutil with quota enforcement
- **Circuit Breakers**: Auto-suspend on repeated violations

### 2. Hook System (16 Integration Points)
- **VM Hooks**: `before_cycle` (100ns), `after_snapshot` (10ms), `on_exception` (1µs)
- **Kernel Hooks**: `syscall_intercept` (1µs), `on_alloc` (500ns), `on_gc` (1ms)
- **Overhead Enforcement**: Callbacks exceeding budget 3x consecutively → forcible detach

### 3. Advanced Type System
- **Linear Types**: `LinearResource` (must consume), `AffineResource` (consume at most once)
- **Dependent Types**: `Vector[N]`, `Range[min,max]`, `NonEmpty[T]`
- **Enforcement**: Metaclass validation at class definition time

### 4. Byzantine Fault Tolerance
- **PBFT Consensus**: Tolerate f < n/3 faulty replicas
- **Hybrid Logical Clocks**: Total order + causality tracking
- **Plumtree Gossip**: Eager push / lazy pull, O(n) message complexity

### 5. Capability Security
- **Macaroon Tokens**: Chained HMAC with attenuating caveats
- **Revocation**: Audit trail with capability metadata logging
- **Sandboxing**: Memory/network namespace isolation

## Testing & Validation

```bash
# Run MVP integration tests
pytest void_state_tools/tests/test_integration.py -v

# Validate hardening (10 core modules)
python validate_hardening.py

# Run benchmarks
pytest void_state_tools/tests/test_benchmarks.py --benchmark-only
```

## Performance Budgets

| Hook Type | Frequency | Budget | Enforcement |
|-----------|-----------|--------|-------------|
| Per-cycle | ~1 GHz | 100ns | Detach after 3 violations |
| Per-instruction | ~1 GHz | 50ns | Statistical sampling |
| Per-event | Variable | 1µs | Immediate detach |
| Per-snapshot | ~1/min | 10ms | Throttle + backpressure |

## Deployment Progress

**Phase 1 (Complete)**: 3/3 tools ✅
- All foundational pattern analysis and entropy measurement tools

**Phase 2 (Active - 40%)**: 6/15 tools 🚧
- Threat detection and behavioral anomaly systems
- Novelty detection and pattern recognition
- Timeline branching and prophecy engines
- External interference detection
- Remaining: 9 advanced analysis tools

**Phase 3 (Active - 17%)**: 4/24 tools 🔬
- Complete meta-tooling system (synthesis, combination, mutation, fitness evaluation)
- Enables recursive self-improvement and automated tool generation
- Remaining: 20 specialized analysis and protocol engineering tools

See [PHASE_2_3_IMPLEMENTATION.md](PHASE_2_3_IMPLEMENTATION.md) for detailed implementation notes and [VOID_STATE_STARTUP_ROADMAP.md](VOID_STATE_STARTUP_ROADMAP.md) for complete phased deployment strategy.

## Project Statistics

- **39 Python modules** (19,598 lines of code)
- **21 markdown docs** (comprehensive architecture, API, deployment guides)
- **10 validated hardening modules** (linear types, HLC, PBFT, capabilities, effects, etc.)
- **7 CI/CD workflows** (testing, linting, publishing, multi-platform)

## License

Proprietary. See [LICENSE](LICENSE) for details.

---

**Questions?** Read the [FAQ](VOID_STATE_FAQ.md) or consult the [Contributing Guide](VOID_STATE_CONTRIBUTING.md).
