"""
Chaos Engineering Service
"The Antichrist Service" - The Final Judgment

A service whose only job is to destroy the system.
It randomly kills processes, corrupts network packets, skews clocks,
and generally tries to break everything.

The goal: Prove the system heals faster than it can be hurt.

Key features:
- Process killing (SIGKILL random services)
- Network chaos (packet loss, delays, corruption)
- Clock skew injection
- Disk I/O corruption
- Resource exhaustion
- Byzantine behavior injection

References:
- "Principles of Chaos Engineering" (Netflix Chaos Monkey)
- "Chaos Engineering" (O'Reilly, 2020)
- Jepsen testing framework
- "Lineage-driven Fault Injection" (Alvaro et al., 2015)
"""

import time
import random
import threading
import signal
import os
from typing import List, Set, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ============================================================================
# CHAOS EXPERIMENT TYPES
# ============================================================================

class ChaosExperimentType(Enum):
    """Types of chaos experiments"""
    PROCESS_KILL = "process_kill"
    NETWORK_DELAY = "network_delay"
    NETWORK_LOSS = "network_loss"
    CLOCK_SKEW = "clock_skew"
    DISK_FULL = "disk_full"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_BURN = "cpu_burn"
    BYZANTINE_MESSAGE = "byzantine_message"


@dataclass
class ChaosExperiment:
    """A chaos experiment to run"""
    experiment_id: str
    experiment_type: ChaosExperimentType
    target: str  # Target node, service, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = False
    error: Optional[str] = None
    
    def __repr__(self):
        return f"{self.experiment_type.value} on {self.target}"


@dataclass
class ChaosResult:
    """Result of a chaos experiment"""
    experiment: ChaosExperiment
    system_survived: bool
    recovery_time: float  # Seconds to recover
    violations_detected: int  # Causality, consistency violations
    message: str


# ============================================================================
# CHAOS ENGINEERING SERVICE
# ============================================================================

class ChaosEngineeringService:
    """
    The Antichrist Service - Tries to destroy the system.
    
    This service continuously injects faults to test system resilience:
    1. Random process kills
    2. Network chaos (delays, losses, corruption)
    3. Clock skew
    4. Resource exhaustion
    5. Byzantine behavior
    
    The system must heal faster than it can be hurt.
    """
    
    def __init__(
        self,
        target_nodes: List[str],
        experiment_interval: float = 10.0,
        max_concurrent_experiments: int = 3
    ):
        """
        Initialize chaos engineering service.
        
        Args:
            target_nodes: Nodes to target with chaos
            experiment_interval: Seconds between experiments
            max_concurrent_experiments: Max experiments running at once
        """
        self.target_nodes = target_nodes
        self.experiment_interval = experiment_interval
        self.max_concurrent_experiments = max_concurrent_experiments
        
        # Experiment state
        self.experiments: List[ChaosExperiment] = []
        self.active_experiments: Set[str] = set()
        self.results: List[ChaosResult] = []
        
        # Statistics
        self.stats = {
            'experiments_run': 0,
            'system_survivals': 0,
            'system_failures': 0,
            'total_chaos_time': 0.0,
            'total_recovery_time': 0.0,
            'violations_detected': 0
        }
        
        # State
        self._lock = threading.Lock()
        self._running = False
        self._chaos_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_experiment_start: Optional[Callable[[ChaosExperiment], None]] = None
        self.on_experiment_complete: Optional[Callable[[ChaosResult], None]] = None
        self.on_system_failure: Optional[Callable[[str], None]] = None
    
    def start(self):
        """Start the chaos engineering service"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._chaos_thread = threading.Thread(target=self._chaos_loop, daemon=True)
            self._chaos_thread.start()
    
    def stop(self):
        """Stop the chaos engineering service"""
        with self._lock:
            self._running = False
        
        if self._chaos_thread:
            self._chaos_thread.join(timeout=5.0)
    
    def _chaos_loop(self):
        """Background loop for chaos injection"""
        while self._running:
            # Run chaos experiment
            self._run_random_experiment()
            
            # Wait before next experiment
            time.sleep(self.experiment_interval)
    
    def _run_random_experiment(self):
        """Run a random chaos experiment"""
        with self._lock:
            # Check if we can run more experiments
            if len(self.active_experiments) >= self.max_concurrent_experiments:
                return
            
            # Select random target
            if not self.target_nodes:
                return
            
            target = random.choice(self.target_nodes)
            
            # Select random experiment type
            experiment_type = random.choice(list(ChaosExperimentType))
            
            # Create experiment
            experiment = ChaosExperiment(
                experiment_id=f"chaos_{int(time.time()*1000)}",
                experiment_type=experiment_type,
                target=target,
                started_at=time.time()
            )
            
            self.experiments.append(experiment)
            self.active_experiments.add(experiment.experiment_id)
            self.stats['experiments_run'] += 1
        
        # Notify callback
        if self.on_experiment_start:
            self.on_experiment_start(experiment)
        
        # Execute experiment
        result = self._execute_experiment(experiment)
        
        # Update state
        with self._lock:
            self.active_experiments.discard(experiment.experiment_id)
            self.results.append(result)
            
            if result.system_survived:
                self.stats['system_survivals'] += 1
            else:
                self.stats['system_failures'] += 1
            
            self.stats['total_chaos_time'] += (experiment.completed_at - experiment.started_at)
            self.stats['total_recovery_time'] += result.recovery_time
            self.stats['violations_detected'] += result.violations_detected
        
        # Notify callback
        if self.on_experiment_complete:
            self.on_experiment_complete(result)
        
        # Check for system failure
        if not result.system_survived:
            if self.on_system_failure:
                self.on_system_failure(result.message)
    
    def _execute_experiment(self, experiment: ChaosExperiment) -> ChaosResult:
        """
        Execute a chaos experiment.
        
        Args:
            experiment: Experiment to execute
            
        Returns:
            Result of the experiment
        """
        start_time = time.time()
        
        try:
            # Execute based on type
            if experiment.experiment_type == ChaosExperimentType.PROCESS_KILL:
                self._inject_process_kill(experiment)
            
            elif experiment.experiment_type == ChaosExperimentType.NETWORK_DELAY:
                self._inject_network_delay(experiment)
            
            elif experiment.experiment_type == ChaosExperimentType.NETWORK_LOSS:
                self._inject_network_loss(experiment)
            
            elif experiment.experiment_type == ChaosExperimentType.CLOCK_SKEW:
                self._inject_clock_skew(experiment)
            
            elif experiment.experiment_type == ChaosExperimentType.DISK_FULL:
                self._inject_disk_full(experiment)
            
            elif experiment.experiment_type == ChaosExperimentType.MEMORY_PRESSURE:
                self._inject_memory_pressure(experiment)
            
            elif experiment.experiment_type == ChaosExperimentType.CPU_BURN:
                self._inject_cpu_burn(experiment)
            
            elif experiment.experiment_type == ChaosExperimentType.BYZANTINE_MESSAGE:
                self._inject_byzantine_message(experiment)
            
            # Mark as complete
            experiment.completed_at = time.time()
            experiment.success = True
            
            # Simulate checking if system survived
            # In production, would check actual system health
            system_survived = random.random() > 0.1  # 90% survival rate
            recovery_time = random.uniform(0.1, 2.0)
            violations = 0 if system_survived else random.randint(1, 5)
            
            return ChaosResult(
                experiment=experiment,
                system_survived=system_survived,
                recovery_time=recovery_time,
                violations_detected=violations,
                message=f"System {'survived' if system_survived else 'FAILED'} {experiment.experiment_type.value}"
            )
        
        except Exception as e:
            experiment.completed_at = time.time()
            experiment.error = str(e)
            
            return ChaosResult(
                experiment=experiment,
                system_survived=False,
                recovery_time=0.0,
                violations_detected=1,
                message=f"Experiment failed: {e}"
            )
    
    def _inject_process_kill(self, experiment: ChaosExperiment):
        """
        Kill a random process.
        
        In production, would actually kill processes.
        For safety, this is simulated.
        """
        # Simulate killing process
        time.sleep(0.1)
        
        # In production:
        # pid = find_target_process(experiment.target)
        # os.kill(pid, signal.SIGKILL)
    
    def _inject_network_delay(self, experiment: ChaosExperiment):
        """
        Inject network delays.
        
        In production, would use tc (traffic control) on Linux:
        tc qdisc add dev eth0 root netem delay 100ms
        """
        delay_ms = random.randint(10, 1000)
        experiment.parameters['delay_ms'] = delay_ms
        time.sleep(0.1)
    
    def _inject_network_loss(self, experiment: ChaosExperiment):
        """
        Inject packet loss.
        
        In production:
        tc qdisc add dev eth0 root netem loss 10%
        """
        loss_pct = random.randint(1, 30)
        experiment.parameters['loss_pct'] = loss_pct
        time.sleep(0.1)
    
    def _inject_clock_skew(self, experiment: ChaosExperiment):
        """
        Skew system clock.
        
        In production, would adjust system time:
        date -s '+1 hour'
        """
        skew_seconds = random.randint(-3600, 3600)
        experiment.parameters['skew_seconds'] = skew_seconds
        time.sleep(0.1)
    
    def _inject_disk_full(self, experiment: ChaosExperiment):
        """
        Fill disk to capacity.
        
        In production:
        dd if=/dev/zero of=/tmp/fill bs=1M count=1000
        """
        time.sleep(0.1)
    
    def _inject_memory_pressure(self, experiment: ChaosExperiment):
        """
        Exhaust memory.
        
        In production, would allocate large memory blocks.
        """
        time.sleep(0.1)
    
    def _inject_cpu_burn(self, experiment: ChaosExperiment):
        """
        Burn CPU cycles.
        
        In production, would spawn CPU-intensive threads.
        """
        time.sleep(0.1)
    
    def _inject_byzantine_message(self, experiment: ChaosExperiment):
        """
        Send Byzantine (malicious) messages.
        
        In production, would send forged consensus messages.
        """
        time.sleep(0.1)
    
    def get_statistics(self) -> dict:
        """Get chaos statistics"""
        with self._lock:
            avg_recovery = (
                self.stats['total_recovery_time'] / 
                max(self.stats['system_survivals'], 1)
            )
            
            survival_rate = (
                (self.stats['system_survivals'] / 
                 max(self.stats['experiments_run'], 1)) * 100
            )
            
            return {
                **self.stats,
                'active_experiments': len(self.active_experiments),
                'total_results': len(self.results),
                'avg_recovery_time': avg_recovery,
                'survival_rate_pct': survival_rate
            }
    
    def get_recent_results(self, count: int = 10) -> List[ChaosResult]:
        """Get recent experiment results"""
        with self._lock:
            return self.results[-count:]


# ============================================================================
# DEMONSTRATION
# ============================================================================

def _example_usage():
    """Demonstrate chaos engineering"""
    
    print("Chaos Engineering Service - The Antichrist")
    print("="*70)
    
    # Create chaos service
    print("\n1. Initializing Chaos Engineering Service...")
    nodes = ["node1", "node2", "node3", "node4"]
    chaos = ChaosEngineeringService(
        target_nodes=nodes,
        experiment_interval=1.0,
        max_concurrent_experiments=2
    )
    
    print(f"   Target nodes: {nodes}")
    print(f"   Experiment interval: {chaos.experiment_interval}s")
    print(f"   Max concurrent: {chaos.max_concurrent_experiments}")
    
    # Set up callbacks
    def on_start(exp: ChaosExperiment):
        print(f"\n   💥 CHAOS: {exp}")
    
    def on_complete(result: ChaosResult):
        status = "✓ SURVIVED" if result.system_survived else "✗ FAILED"
        print(f"   → {status} (recovery: {result.recovery_time:.2f}s)")
    
    def on_failure(message: str):
        print(f"\n   🔥 SYSTEM FAILURE: {message}")
    
    chaos.on_experiment_start = on_start
    chaos.on_experiment_complete = on_complete
    chaos.on_system_failure = on_failure
    
    # Run chaos for a few seconds
    print("\n2. Starting chaos injection...")
    print("   The datacenter lights flicker...")
    
    chaos.start()
    time.sleep(5.0)
    chaos.stop()
    
    # Show statistics
    print("\n3. Chaos statistics:")
    stats = chaos.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n4. Recent experiments:")
    results = chaos.get_recent_results(5)
    for result in results:
        status = "✓" if result.system_survived else "✗"
        print(f"   {status} {result.experiment.experiment_type.value}")
    
    print("\n5. Chaos engineering principles:")
    print("   ✓ Randomly kill processes")
    print("   ✓ Corrupt network packets")
    print("   ✓ Skew clocks")
    print("   ✓ Fill disks")
    print("   ✓ Exhaust memory")
    print("   ✓ Inject Byzantine messages")
    
    print("\n6. The Dream:")
    print("   You turn on the Antichrist.")
    print("   The data center lights flicker.")
    print("   Hard drives fail. Cables are cut.")
    print("   And yet, 'The Uncaused Light' remains.")
    print("   The system heals faster than it can be hurt.")
    print("   Uptime: 100%")
    
    print("\n" + "="*70)
    print("Chaos is the ultimate test of resilience.")


if __name__ == "__main__":
    _example_usage()
