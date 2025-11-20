# Void System Hardening - Architecture Documentation

## Overview

This document describes the architectural improvements made to harden the Void system, transforming it from a proof-of-concept to a production-ready distributed computing platform.

## Table of Contents

1. [Core Type System](#core-type-system)
2. [Distributed Systems](#distributed-systems)
3. [Security](#security)
4. [Effects System](#effects-system)
5. [Advanced Algorithms](#advanced-algorithms)
6. [System Integration](#system-integration)
7. [Performance Considerations](#performance-considerations)

## Core Type System

### Linear Types (`linear_types.py`)

Linear types ensure resources are consumed exactly once, preventing resource leaks in complex control flows.

**Key Features:**
- `LinearResource[T]` - Must be consumed exactly once
- `AffineResource[T]` - Can be used at most once
- `LinearContext` - Tracks resources in a scope
- Automatic leak detection via `__del__`

**Example:**
```python
from linear_types import LinearResource

# Create a linear resource
db_conn = LinearResource(database_connection, name="db_conn")

# Must consume it
conn = db_conn.consume()

# Trying to use again raises LinearTypeError
# conn2 = db_conn.consume()  # ERROR!
```

**Use Cases:**
- Database connections
- File handles
- Network sockets
- Capability tokens
- Cryptographic keys

### Dependent Types (`dependent_types.py`)

Dependent types allow types to depend on values, enabling compile-time/import-time verification of constraints.

**Key Features:**
- `Vector[N]` - Vector of exactly N elements
- `Range[min, max]` - Integer in range [min, max]
- `NonEmpty[T]` - Non-empty collection
- `Refined[T, predicate]` - Value satisfying predicate
- Type caching for performance

**Example:**
```python
from dependent_types import Vector, Range, NonEmpty

# Vector of exactly 3 elements
v = Vector[3]([1, 2, 3])  # OK
# v = Vector[3]([1, 2])    # TypeError!

# Integer in range
age = Range[0, 120](25)    # OK
# age = Range[0, 120](150) # TypeError!

# Non-empty list
items = NonEmpty[list]([1, 2, 3])  # OK
# items = NonEmpty[list]([])        # TypeError!
```

**Use Cases:**
- Array/vector size verification
- Bounds checking
- Non-null guarantees
- Configuration validation

## Distributed Systems

### Hybrid Logical Clocks (`hlc.py`)

HLC combines physical time (NTP) with logical counters to provide causality-preserving timestamps.

**Key Features:**
- `HLCTimestamp` - Physical + logical + node_id
- `HybridLogicalClock` - Thread-safe clock
- `LWWRegister` - Last-Write-Wins CRDT
- `HLCVersionStore` - Multi-version concurrency control

**Example:**
```python
from hlc import HybridLogicalClock

clock_a = HybridLogicalClock("node_a")
clock_b = HybridLogicalClock("node_b")

# Local event
ts1 = clock_a.now()

# Message passing (A → B)
ts2 = clock_b.update(ts1)  # Updates B's clock

# Causality guaranteed
assert ts1 < ts2
```

**Advantages over Alternatives:**
- Better than wall-clock time (no drift issues)
- Better than pure logical clocks (human-readable)
- Total ordering (unlike vector clocks)

### Plumtree Gossip (`gossip.py`)

Epidemic broadcast trees optimize gossip dissemination from O(n²) to O(n).

**Key Features:**
- `PlumtreeNode` - Implements Plumtree protocol
- Eager push (spanning tree)
- Lazy push (gossip fallback)
- PRUNE/GRAFT optimization
- Self-healing

**Example:**
```python
from gossip import PlumtreeNode

node = PlumtreeNode("node_id")
node.add_peer("peer1", eager=True)
node.add_peer("peer2", eager=False)

# Broadcast message
msg_id = node.broadcast({"data": "hello"})

# Automatic optimization via PRUNE on duplicates
```

**Performance:**
- Message complexity: O(n) instead of O(n²)
- Fast dissemination via spanning tree
- Reliability via lazy gossip

### PBFT Consensus (`pbft.py`)

Practical Byzantine Fault Tolerance provides consensus with Byzantine fault tolerance.

**Key Features:**
- `PBFTNode` - PBFT protocol implementation
- Three-phase commit (pre-prepare, prepare, commit)
- View changes for fault tolerance
- HLC timestamp integration
- Ready for Rust migration

**Example:**
```python
from pbft import PBFTNode

# Create 4 replicas (tolerates f=1 failures)
node = PBFTNode("node_id", ["node0", "node1", "node2", "node3"], f=1)

# Submit request (if primary)
node.request({"op": "write", "key": "x", "value": 10})

# Consensus reached after 2f+1 prepares and commits
```

**Guarantees:**
- Safety: Never returns incorrect result
- Liveness: Eventually makes progress
- Byzantine tolerance: f < n/3

## Security

### Capability-Based Security (`capabilities.py`)

Macaroons provide decentralized authorization with token attenuation.

**Key Features:**
- `Macaroon` - Chained HMAC token
- `MacaroonFactory` - Mints and verifies tokens
- `CapabilityToken` - Linear capability
- `CapabilityManager` - Enforces consumption
- Caveat verifiers (time, user, IP)

**Example:**
```python
from capabilities import MacaroonFactory, CapabilityManager, create_standard_verifier

factory = MacaroonFactory(root_key)
verifier = create_standard_verifier()
manager = CapabilityManager(factory, verifier)

# Grant capability
cap = manager.grant_capability(
    "/api/data",
    CapabilityType.READ,
    caveats=[f"time < {future_time}"]
)

# Verify and consume (linear!)
valid = manager.verify_capability(cap, context)
```

**Advantages:**
- Decentralized attenuation
- No server roundtrips
- Contextual caveats
- Linear type integration

## Effects System

### Algebraic Effects (`effects.py`)

Algebraic effects provide dependency injection, testability, and decoupling.

**Key Features:**
- Effect definitions (time, network, state, log, random)
- Effect handlers (real vs mock)
- Thread-local handler stack
- Deterministic testing

**Example:**
```python
from effects import with_handlers, TimeHandler, LogHandler, ask_current_time

# Production mode
with with_handlers(TimeHandler("real"), LogHandler("print")):
    t = ask_current_time()  # Returns time.time()

# Testing mode
time_handler = TimeHandler("mock")
time_handler.set_mock_time(1000.0)

with with_handlers(time_handler):
    t = ask_current_time()  # Returns 1000.0
```

**Use Cases:**
- Deterministic testing of race conditions
- Network mocking
- Time travel for temporal logic
- State management without globals

## Advanced Algorithms

### Quantum-Inspired Scheduling (`quantum_scheduling.py`)

Uses the Ising model to optimize task placement via energy minimization.

**Key Features:**
- `IsingScheduler` - Simulated annealing solver
- Hamiltonian components (overload, communication, imbalance)
- Customizable penalty weights
- Load balancing optimization

**Example:**
```python
from quantum_scheduling import IsingScheduler, Task, Node

tasks = [
    Task("task1", {"cpu": 2.0, "memory": 4.0}),
    Task("task2", {"cpu": 1.0, "memory": 2.0})
]

nodes = [
    Node("node1", {"cpu": 4.0, "memory": 8.0}),
    Node("node2", {"cpu": 4.0, "memory": 8.0})
]

scheduler = IsingScheduler(tasks, nodes)
schedule = scheduler.schedule()
# Returns: {"task1": "node1", "task2": "node2"}
```

**Optimization Goals:**
- Minimize node overload
- Minimize communication cost (co-locate communicating tasks)
- Balance load across nodes

### Causal Inference (`causal.py`)

Structural causal models for root cause analysis using do-calculus.

**Key Features:**
- `CausalGraph` - Static topology with learned weights
- `CausalInference` - Do-calculus queries
- Root cause detection
- Failure propagation learning

**Example:**
```python
from causal import CausalGraph, CausalInference

# Build system topology
graph = CausalGraph()
graph.add_edge("Database", "Cache", 0.7)
graph.add_edge("Cache", "API", 0.8)

inference = CausalInference(graph)

# Observational query
p_obs = inference.observational_probability("API", {"Cache": True})

# Interventional query (do-calculus)
p_int = inference.interventional_probability("API", {"Cache": True})

# Root cause analysis
failed = {"Database", "Cache", "API"}
root_causes = inference.find_root_causes(failed)
```

**Distinction:**
- P(Y | X) - Observational (correlation)
- P(Y | do(X)) - Interventional (causation)

## System Integration

### System Orchestrator (`system.py`)

Main entry point that boots all subsystems in the correct order.

**Boot Sequence:**
1. Type system initialization (metaclasses)
2. Effect system setup
3. Distributed systems (HLC, gossip, PBFT)
4. Security (capabilities)
5. Scheduling and algorithms
6. Monitoring and observability

**Example:**
```python
from system import VoidSystem, SystemConfig

config = SystemConfig(
    node_id="void-primary",
    enable_gossip=True,
    enable_pbft=True,
    replica_ids=["node0", "node1", "node2", "node3"],
    pbft_f=1
)

system = VoidSystem(config)
system.initialize()
system.run()  # Enters main event loop
```

**Features:**
- Lifecycle management
- Graceful shutdown
- Status reporting
- Signal handling

## Performance Considerations

### Current State: Python

All modules are implemented in Python for:
- Correctness and clarity
- Rapid iteration
- Easy debugging

### Future State: Rust Core

Performance-critical components should be migrated to Rust:

**High Priority:**
1. **PBFT consensus**
   - Ed25519 signatures: 100x faster
   - Message serialization: 10-50x faster
   - Overall consensus: 50-100x faster

2. **Quantum scheduling**
   - Simulated annealing loops: 100x faster
   - Energy computation: 50x faster
   - Overall scheduling: 50-100x faster

3. **Cryptography (Macaroons)**
   - HMAC operations: 10-20x faster
   - Signature verification: 100x faster

**Architecture:**
- Python = "control plane" (orchestration, high-level logic)
- Rust = "data plane" (crypto, message passing, compute)
- PyO3 for seamless integration

### Recommended Next Steps

1. Set up Rust toolchain (cargo, rustc)
2. Configure Maturin for Python bindings
3. Port PBFT cryptographic operations to Rust
4. Port quantum scheduling loops to Rust
5. Benchmark and validate performance gains

### Expected Performance

With Rust core:
- **Consensus throughput**: 1,000 → 100,000 ops/sec
- **Scheduling latency**: 100ms → 1ms
- **Message overhead**: 10MB/sec → 500MB/sec

## Testing

All modules include demonstration/testing code:

```bash
# Test individual modules
python3 linear_types.py
python3 dependent_types.py
python3 hlc.py
python3 gossip.py
python3 capabilities.py
python3 effects.py
python3 quantum_scheduling.py
python3 causal.py
python3 pbft.py

# Test system integration
python3 system.py
```

## References

### Papers
- Castro & Liskov (1999): "Practical Byzantine Fault Tolerance"
- Kulkarni et al. (2014): "Logical Physical Clocks"
- Leitão et al. (2007): "Epidemic Broadcast Trees"
- Birgisson et al. (2014): "Macaroons"
- Pearl (2009): "Causality"
- Lucas (2014): "Ising formulations of many NP problems"

### Books
- Tanenbaum & Van Steen: "Distributed Systems"
- Kleppmann: "Designing Data-Intensive Applications"
- Cachin et al.: "Introduction to Reliable and Secure Distributed Programming"

## Conclusion

The Void system now has a solid architectural foundation:

1. **Type Safety**: Linear and dependent types prevent errors
2. **Distributed Consensus**: HLC + PBFT for Byzantine tolerance
3. **Secure Authorization**: Macaroon capabilities
4. **Testability**: Algebraic effects enable deterministic testing
5. **Smart Scheduling**: Quantum-inspired optimization
6. **Root Cause Analysis**: Causal inference for debugging

The system is designed for eventual Rust migration while maintaining Python as the control plane.
