"""
Chaos Engineering Integration

Introduces controlled failures to test system resilience.
Based on Chaos Monkey and related tools.
"""

from typing import List, Callable, Any
import random

class ChaosExperiment:
    """Definition of a chaos experiment"""
    def __init__(self, name: str, fault_injector: Callable, probability: float = 0.1):
        self.name = name
        self.fault_injector = fault_injector
        self.probability = probability
        
    def maybe_inject(self):
        """Randomly inject fault based on probability"""
        if random.random() < self.probability:
            self.fault_injector()

class ChaosEngine:
    """Engine for chaos engineering experiments"""
    def __init__(self):
        self.experiments: List[ChaosExperiment] = []
        self.enabled = False
        
    def register_experiment(self, experiment: ChaosExperiment):
        """Register chaos experiment"""
        self.experiments.append(experiment)
    
    def enable(self):
        """Enable chaos experiments"""
        self.enabled = True
    
    def disable(self):
        """Disable chaos experiments"""
        self.enabled = False
    
    def inject_chaos(self):
        """Inject chaos faults"""
        if self.enabled:
            for exp in self.experiments:
                exp.maybe_inject()

# Common fault injection patterns
def inject_latency(delay_ms: float = 100):
    """Inject network latency"""
    import time
    time.sleep(delay_ms / 1000.0)

def inject_exception():
    """Inject random exception"""
    raise RuntimeError("Chaos engineering: Injected failure")

def inject_resource_exhaustion():
    """Simulate resource exhaustion"""
    pass  # Would allocate large memory or CPU
