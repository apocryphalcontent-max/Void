# Void: Self-Aware AI System Architecture

> **A layered, capability-based framework for AI system introspection, maintenance, education, mutation, and defense.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-MVP%20Complete-green.svg)]()

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

## Current Status: MVP Complete

**Version:** 1.0.0-mvp-complete
**Phase 1 Tools (3/3):**
- ✅ **PatternPrevalenceQuantifier**: Tracks pattern frequency and ubiquity across system state
- ✅ **LocalEntropyMicroscope**: Measures Shannon entropy at microscopic scales with gradient analysis
- ✅ **EventSignatureClassifier**: Naive Bayes classifier for event taxonomic categorization

**Infrastructure (Complete):**
- Tool Registry & Lifecycle Manager (DORMANT → ACTIVE → SUSPENDED → TERMINATED)
- VM/Kernel Hook System (16 hook points with overhead budgets)
- Linear & Dependent Type System (resource safety, dimensionality checking)
- Distributed Consensus (HLC, Plumtree gossip, PBFT f<n/3 fault tolerance)
- Capability System (chained macaroon tokens with caveats)

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Import and use MVP tools
from void_state_tools import (
    ToolRegistry, ToolConfig,
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier
)

# Create registry and register a tool
registry = ToolRegistry()
config = ToolConfig(
    tool_name="pattern_analyzer",
    max_memory_mb=100,
    max_cpu_percent=10,
    overhead_budget_ns=1000
)

tool = PatternPrevalenceQuantifier(config)
handle = registry.register_tool(tool)
registry.lifecycle_manager.attach_tool(handle.tool_id)

# Analyze patterns
result = tool.analyze({
    "pattern": "memory_spike",
    "context": "inference_loop",
    "timestamp": time.time()
})
print(f"Frequency: {result['frequency_ratio']:.2%}")
print(f"Contexts: {result['context_diversity']}")
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

## Future Phases

**Phase 2 (Months 7-18)**: Timeline Branching, Prophecy Engine, Threat Signature Recognizer

**Phase 3 (Months 19-36)**: Tool Synthesizer, Protocol Genome Analyzer, Recursive Meta-Tool

See [VOID_STATE_STARTUP_ROADMAP.md](VOID_STATE_STARTUP_ROADMAP.md) for complete phased deployment strategy.

## Project Statistics

- **39 Python modules** (19,598 lines of code)
- **21 markdown docs** (comprehensive architecture, API, deployment guides)
- **10 validated hardening modules** (linear types, HLC, PBFT, capabilities, effects, etc.)
- **7 CI/CD workflows** (testing, linting, publishing, multi-platform)

## License

Proprietary. See [LICENSE](LICENSE) for details.

---

**Questions?** Read the [FAQ](VOID_STATE_FAQ.md) or consult the [Contributing Guide](VOID_STATE_CONTRIBUTING.md).
