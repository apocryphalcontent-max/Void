# Void-State Tools API Documentation

## Core Concepts

### Tool Lifecycle

Tools in the Void-State system follow a strict lifecycle:

```
DORMANT → INITIALIZING → ACTIVE ⇄ SUSPENDED → TERMINATED
```

- **DORMANT**: Tool is registered but not initialized
- **INITIALIZING**: Tool is being initialized (transient state)
- **ACTIVE**: Tool is running and processing events
- **SUSPENDED**: Tool is temporarily paused
- **TERMINATED**: Tool has been shut down

### Tool Types

#### Base Tool
All tools inherit from the `Tool` base class:

```python
from void_state_tools import Tool, ToolConfig

class MyTool(Tool):
    def initialize(self) -> bool:
        """Initialize resources. Return True on success."""
        pass
    
    def shutdown(self) -> bool:
        """Cleanup resources. Return True on success."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata."""
        return {
            "name": "My Tool",
            "category": "custom",
            "version": "1.0.0"
        }
```

#### AnalysisTool
For tools that analyze data:

```python
from void_state_tools.base import AnalysisTool

class MyAnalyzer(AnalysisTool):
    def analyze(self, data: Any) -> Any:
        """Analyze data and return results."""
        pass
```

#### MonitoringTool
For tools that monitor system state:

```python
from void_state_tools.base import MonitoringTool

class MyMonitor(MonitoringTool):
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current metrics."""
        pass
```

### Tool Registration

```python
from void_state_tools import ToolRegistry, ToolConfig

# Create registry
registry = ToolRegistry()

# Configure tool
config = ToolConfig(
    tool_name="my_tool",
    tool_category="custom",
    max_memory_mb=100,
    max_cpu_percent=10
)

# Create and register tool
tool = MyTool(config)
handle = registry.register_tool(tool)

# Attach (initialize and activate)
registry.lifecycle_manager.attach_tool(handle.tool_id)
```

### Hook System

#### Registering Hook Callbacks

```python
from void_state_tools.hooks import HookRegistry

hook_registry = HookRegistry()

# Get a specific hook
hook = hook_registry.get_hook("vm.after_cycle")

# Register callback
def my_callback(context):
    print(f"Cycle {context.cycle_count} completed")

hook.register(my_callback, priority=50)
```

#### Hook Filtering

```python
from void_state_tools.hooks import FrequencyFilter, ConditionalFilter

# Sample 10% of events
frequency_filter = FrequencyFilter(sample_rate=0.1)

# Conditional filtering
conditional_filter = ConditionalFilter(
    lambda ctx: ctx.cycle_count % 100 == 0
)

hook.register(
    my_callback,
    filter_fn=frequency_filter.should_trigger
)
```

## API Reference

### ToolConfig

```python
ToolConfig(
    tool_id: Optional[str] = None,
    tool_name: str = "unnamed_tool",
    tool_category: str = "uncategorized",
    max_memory_mb: int = 100,
    max_cpu_percent: int = 10,
    max_io_ops_per_second: int = 1000,
    overhead_budget_ns: int = 1000,
    enabled: bool = True,
    auto_start: bool = True,
    priority: int = 0
)
```

### ToolRegistry Methods

```python
# Register a tool
handle = registry.register_tool(tool: Tool) -> ToolHandle

# Get a tool by ID
tool = registry.get_tool(tool_id: str) -> Optional[ToolHandle]

# List all tools
tools = registry.list_tools(
    category: Optional[str] = None,
    state: Optional[ToolState] = None
) -> List[ToolHandle]

# Find tools with custom predicate
tools = registry.find_tools(
    predicate: Callable[[ToolHandle], bool]
) -> List[ToolHandle]

# Unregister a tool
success = registry.unregister_tool(tool_id: str) -> bool
```

### ToolLifecycleManager Methods

```python
# Attach (initialize and activate) a tool
success = lifecycle_manager.attach_tool(tool_id: str) -> bool

# Detach (shutdown) a tool
success = lifecycle_manager.detach_tool(tool_id: str) -> bool

# Suspend a tool
success = lifecycle_manager.suspend_tool(tool_id: str) -> bool

# Resume a suspended tool
success = lifecycle_manager.resume_tool(tool_id: str) -> bool

# Force detach (emergency shutdown)
success = lifecycle_manager.force_detach_tool(tool_id: str) -> bool

# Get tool state
state = lifecycle_manager.get_tool_state(tool_id: str) -> ToolState
```

### HookRegistry Methods

```python
# Get a hook by name
hook = hook_registry.get_hook(name: str) -> Optional[HookPoint]

# List all hooks
hooks = hook_registry.list_hooks(
    prefix: Optional[str] = None
) -> List[str]

# Check if hook exists
exists = hook_registry.has_hook(name: str) -> bool
```

### HookPoint Methods

```python
# Register callback
reg_id = hook.register(
    callback: Callable[[HookContext], Any],
    priority: int = 0,
    filter_fn: Optional[Callable[[HookContext], bool]] = None
) -> str

# Unregister callback
success = hook.unregister(registration_id: str) -> bool

# Execute all callbacks
results = hook.execute(context: HookContext) -> List[Any]

# Clear all callbacks
hook.clear()
```

## Examples

### Example 1: Custom Memory Monitor

```python
from void_state_tools import Tool, ToolConfig, ToolRegistry
from void_state_tools.hooks import HookRegistry

class MemoryMonitor(Tool):
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.peak_memory = 0
    
    def initialize(self) -> bool:
        hook_registry = HookRegistry()
        hook = hook_registry.get_hook("kernel.on_memory_allocation")
        hook.register(self.on_allocation)
        return True
    
    def on_allocation(self, context):
        size = context.additional_data.get("size", 0)
        if size > self.peak_memory:
            self.peak_memory = size
            print(f"New peak memory: {size} bytes")
    
    def get_metadata(self):
        return {
            "name": "Memory Monitor",
            "category": "monitoring",
            "version": "1.0.0"
        }

# Usage
registry = ToolRegistry()
monitor = MemoryMonitor(ToolConfig())
handle = registry.register_tool(monitor)
registry.lifecycle_manager.attach_tool(handle.tool_id)
```

### Example 2: Anomaly Detection Pipeline

```python
from void_state_tools.mvp import StatisticalAnomalyDetector
from void_state_tools import ToolRegistry, ToolConfig

# Create detector
config = ToolConfig(
    tool_name="anomaly_pipeline",
    max_memory_mb=200
)
detector = StatisticalAnomalyDetector(config)

# Register and activate
registry = ToolRegistry()
handle = registry.register_tool(detector)
registry.lifecycle_manager.attach_tool(handle.tool_id)

# Detector is now monitoring for anomalies
print(f"Detector active: {handle.state}")
```

### Example 3: Multi-Tool Coordination

```python
from void_state_tools import ToolRegistry
from void_state_tools.mvp import (
    StructuralMemoryDiffAnalyzer,
    StatisticalAnomalyDetector,
    ExecutionLineageTracer
)

registry = ToolRegistry()

# Register multiple tools
tools = [
    StructuralMemoryDiffAnalyzer(ToolConfig(priority=100)),
    StatisticalAnomalyDetector(ToolConfig(priority=90)),
    ExecutionLineageTracer(ToolConfig(priority=80))
]

handles = []
for tool in tools:
    handle = registry.register_tool(tool)
    handles.append(handle)
    registry.lifecycle_manager.attach_tool(handle.tool_id)

# All tools now running in priority order
print(f"Active tools: {len(registry.list_tools())}")
```

## Best Practices

1. **Resource Management**: Always set appropriate resource limits
2. **Hook Priorities**: Use priorities to control execution order
3. **Filtering**: Use filters to reduce overhead on high-frequency hooks
4. **Error Handling**: Implement robust error handling in callbacks
5. **Metrics**: Monitor tool performance regularly
6. **Testing**: Test tools in isolation before deployment

## Performance Guidelines

- **Per-cycle hooks**: < 100ns overhead budget
- **Per-event hooks**: < 1µs overhead budget
- **Per-snapshot hooks**: < 10ms overhead budget
- **Tool memory**: < 100MB default, adjust as needed
- **Tool CPU**: < 10% default, adjust as needed

## Troubleshooting

### Tool Won't Initialize
- Check resource availability
- Verify dependencies are met
- Review error logs

### High Overhead
- Reduce hook callback complexity
- Use filtering to reduce execution frequency
- Lower tool priority

### Memory Leaks
- Ensure cleanup in shutdown()
- Monitor metrics regularly
- Use memory profilers

For more information, see the full documentation at `VOID_STATE_TOOLS_README.md`.
