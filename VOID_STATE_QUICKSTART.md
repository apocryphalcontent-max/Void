# Void-State Tools: Quick Start Guide

## Get Started in 5 Minutes

### Installation

```bash
git clone https://github.com/apocryphalcontent-max/Messy.git
cd Messy
pip install -e .
```

### Your First Tool

```python
from void_state_tools import Tool, ToolConfig, ToolRegistry

class HelloTool(Tool):
    def initialize(self) -> bool:
        print("Hello from Void-State!")
        return True
    
    def shutdown(self) -> bool:
        print("Goodbye!")
        return True
    
    def suspend(self) -> bool:
        return True
    
    def resume(self) -> bool:
        return True
    
    def get_metadata(self):
        return {
            "name": "Hello Tool",
            "category": "examples",
            "version": "1.0.0"
        }

# Use it
registry = ToolRegistry()
tool = HelloTool(ToolConfig())
handle = registry.register_tool(tool)
registry.lifecycle_manager.attach_tool(handle.tool_id)
# Prints: Hello from Void-State!
```

### Use MVP Tools

```python
from void_state_tools.mvp import (
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope
)

# Pattern analysis
quantifier = PatternPrevalenceQuantifier(ToolConfig())
quantifier.initialize()

result = quantifier.analyze({
    "pattern": "user_login",
    "context": "web_app",
    "timestamp": time.time()
})

print(f"Frequency: {result['frequency']}")
print(f"Context diversity: {result['context_diversity']}")
print(f"Is common: {result['is_common']}")

# Entropy monitoring
entropy = LocalEntropyMicroscope(ToolConfig())
entropy.initialize()

result = entropy.observe_region("cache", {"key1": "value1"})
print(f"Entropy: {result['entropy']:.2f}")
print(f"Is stable: {result['is_stable']}")
```

### Monitor with Prometheus

```python
from void_state_tools.monitoring import start_metrics_server

# Start metrics server
start_metrics_server(port=9090)

# Access metrics at: http://localhost:9090/metrics
# Health check at: http://localhost:9090/health
```

### Run Benchmarks

```bash
python -m void_state_tools.benchmarks
```

Output:
```
Benchmarking hook overhead (10000 iterations)...

Hook Overhead:
  Mean: 87 ns | P95: 145 ns | P99: 198 ns
  Overhead: 87.0% ✓ PASS
```

### Run Tests

```bash
pytest void_state_tools/tests/ -v
```

### Deploy with Docker

```bash
docker build -t void-state-tools .
docker run -p 9090:9090 void-state-tools
```

### Next Steps

- Read the [API Documentation](void_state_tools/docs/API.md)
- Check [Deployment Guide](VOID_STATE_DEPLOYMENT_GUIDE.md)
- Review [Startup Roadmap](VOID_STATE_STARTUP_ROADMAP.md)
- Explore [Phase 2 Tools](void_state_tools/phase2/)

## Common Use Cases

### Anomaly Detection

```python
from void_state_tools.mvp import StatisticalAnomalyDetector

detector = StatisticalAnomalyDetector(ToolConfig())
detector.initialize()

# Detector automatically monitors via hooks
# Check metrics:
metrics = detector.collect_metrics()
print(f"Anomalies detected: {metrics.get('anomalies_total', 0)}")
```

### Pattern Tracking

```python
from void_state_tools.mvp import PatternPrevalenceQuantifier

quantifier = PatternPrevalenceQuantifier(ToolConfig())
quantifier.initialize()

# Track multiple patterns
patterns = ["api_call", "db_query", "cache_hit"]
for pattern in patterns:
    quantifier.analyze({"pattern": pattern, "context": "production"})

# Get top patterns
top = quantifier.get_top_patterns(n=5)
for item in top:
    print(f"{item['pattern']}: {item['frequency']} ({item['frequency_ratio']:.2%})")
```

### Execution Tracing

```python
from void_state_tools.mvp import ExecutionLineageTracer

tracer = ExecutionLineageTracer(ToolConfig())
tracer.initialize()

# Tracer automatically captures execution via hooks
# Access trace data through tool interface
```

## Troubleshooting

**Import Error:**
```bash
pip install -e .
```

**Tests Fail:**
```bash
pip install pytest pytest-cov
pytest void_state_tools/tests/ -v
```

**Metrics Not Available:**
```python
from void_state_tools.monitoring import start_metrics_server
start_metrics_server(port=9090)
```

## Getting Help

- Check documentation in `VOID_STATE_TOOLS_README.md`
- Review API docs in `void_state_tools/docs/API.md`
- See deployment guide in `VOID_STATE_DEPLOYMENT_GUIDE.md`
- Look at configuration examples in `void_state_tools/config_examples/`

## What's Next?

Phase 1 MVP (6 tools) is complete. Phase 2 (15 tools) includes:
- Timeline Branching Engine
- Prophecy Engine  
- Threat Signature Recognizer
- Semantic Memory Diff Analyzer
- And 11 more advanced tools

See `VOID_STATE_STARTUP_ROADMAP.md` for the complete roadmap.
