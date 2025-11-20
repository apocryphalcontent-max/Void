"""
Differential Privacy for Metrics

Adds differential privacy to published metrics to protect sensitive information.
Provides mathematical guarantee of privacy while preserving utility.
"""

import numpy as np
from typing import List, Callable, Dict
from dataclasses import dataclass

@dataclass
class PrivacyBudget:
    """Privacy budget (epsilon, delta) for differential privacy"""
    epsilon: float  # Privacy parameter (smaller = stronger privacy)
    delta: float = 1e-6  # Failure probability
    
    def spend(self, epsilon_cost: float):
        """Spend privacy budget"""
        if epsilon_cost > self.epsilon:
            raise ValueError("Insufficient privacy budget")
        self.epsilon -= epsilon_cost
    
    def remaining(self) -> float:
        """Get remaining privacy budget"""
        return self.epsilon

class DifferentiallyPrivateMetric:
    """
    Apply differential privacy to metric publication.
    
    Uses Laplace mechanism for numerical queries.
    """
    def __init__(self, epsilon: float, sensitivity: float):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
    def add_noise(self, true_value: float) -> float:
        """
        Add calibrated Laplace noise to achieve differential privacy.
        
        Laplace scale: sensitivity / epsilon
        """
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale)
        return true_value + noise
    
    def query(self, query_function: Callable[[], float]) -> float:
        """Execute query and return differentially private result"""
        true_result = query_function()
        private_result = self.add_noise(true_result)
        return private_result

class GaussianMechanism:
    """Gaussian mechanism for (epsilon, delta)-differential privacy"""
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()
        
    def _compute_sigma(self) -> float:
        """Compute sigma for Gaussian noise"""
        import math
        return (self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
    
    def add_noise(self, true_value: float) -> float:
        """Add Gaussian noise"""
        noise = np.random.normal(loc=0, scale=self.sigma)
        return true_value + noise

class PrivacyAccountant:
    """Track cumulative privacy loss under composition"""
    def __init__(self, total_budget: PrivacyBudget):
        self.total_budget = total_budget
        self.queries = []
        
    def account_query(self, epsilon: float, delta: float = 0):
        """Record privacy cost of query"""
        self.queries.append((epsilon, delta))
        
    def get_total_privacy_loss(self) -> tuple:
        """Compute total privacy loss using advanced composition"""
        if not self.queries:
            return (0.0, 0.0)
        
        k = len(self.queries)
        epsilons = [e for e, d in self.queries]
        deltas = [d for e, d in self.queries]
        
        epsilon_max = max(epsilons)
        delta_total = sum(deltas) + self.total_budget.delta / 2
        epsilon_total = np.sqrt(2 * k * np.log(1/delta_total)) * epsilon_max
        
        return (epsilon_total, delta_total)

# Applications:
# - Private metric publication
# - Sensitive data protection
# - Statistical database queries
# - Privacy-preserving ML
