"""
Quantum-Inspired Scheduling using Ising Model

Implements task scheduling using the Ising model from statistical physics,
where task placement is optimized by minimizing system energy.

The Ising model maps scheduling constraints to:
- Energy minimization (optimal placement)
- Spin configurations (task assignments)
- Hamiltonian (cost function)

This enables:
- Load balancing across nodes
- Locality-aware placement (minimize communication)
- Constraint satisfaction (resource limits)

References:
- "Quantum Annealing for Combinatorial Optimization" (Lucas, 2014)
- D-Wave's QUBO formulations
- "Ising formulations of many NP problems" (Lucas, 2014)
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import random
import math


# ============================================================================
# TASK AND NODE DEFINITIONS
# ============================================================================

@dataclass
class Task:
    """A task to be scheduled"""
    task_id: str
    resource_requirements: Dict[str, float]  # CPU, memory, etc.
    communication_partners: Set[str] = None  # Other tasks it communicates with
    communication_cost: Dict[str, float] = None  # Cost per partner
    
    def __post_init__(self):
        if self.communication_partners is None:
            self.communication_partners = set()
        if self.communication_cost is None:
            self.communication_cost = {}
    
    def __hash__(self):
        return hash(self.task_id)


@dataclass
class Node:
    """A compute node"""
    node_id: str
    capacity: Dict[str, float]  # Max CPU, memory, etc.
    current_load: Dict[str, float] = None  # Current usage
    location: Optional[Tuple[float, float]] = None  # Physical location
    
    def __post_init__(self):
        if self.current_load is None:
            self.current_load = {k: 0.0 for k in self.capacity.keys()}
    
    def available_capacity(self, resource: str) -> float:
        """Get available capacity for resource"""
        return self.capacity.get(resource, 0.0) - self.current_load.get(resource, 0.0)
    
    def can_accommodate(self, task: Task) -> bool:
        """Check if node can accommodate task"""
        for resource, required in task.resource_requirements.items():
            if self.available_capacity(resource) < required:
                return False
        return True
    
    def __hash__(self):
        return hash(self.node_id)


# ============================================================================
# ISING MODEL SCHEDULER
# ============================================================================

class IsingScheduler:
    """
    Task scheduler using Ising model energy minimization.
    
    The scheduling problem is formulated as finding spin configurations
    that minimize the Hamiltonian (energy function).
    
    Hamiltonian components:
    1. Overload penalty: Penalize nodes exceeding capacity
    2. Communication penalty: Penalize distant task placement
    3. Balance penalty: Encourage even load distribution
    """
    
    def __init__(
        self,
        tasks: List[Task],
        nodes: List[Node],
        penalty_overload: float = 100.0,
        penalty_communication: float = 10.0,
        penalty_imbalance: float = 5.0
    ):
        """
        Initialize Ising scheduler.
        
        Args:
            tasks: Tasks to schedule
            nodes: Available nodes
            penalty_overload: Weight for overload penalty
            penalty_communication: Weight for communication penalty
            penalty_imbalance: Weight for load imbalance penalty
        """
        self.tasks = tasks
        self.nodes = nodes
        self.penalty_overload = penalty_overload
        self.penalty_communication = penalty_communication
        self.penalty_imbalance = penalty_imbalance
        
        # Create mapping indices
        self.task_idx = {t.task_id: i for i, t in enumerate(tasks)}
        self.node_idx = {n.node_id: i for i, n in enumerate(nodes)}
        
        # Assignment matrix: assignment[i][j] = 1 if task i on node j
        self.num_tasks = len(tasks)
        self.num_nodes = len(nodes)
    
    def compute_energy(self, assignment: NDArray[np.int32]) -> float:
        """
        Compute Hamiltonian (energy) for an assignment.
        
        Lower energy = better assignment.
        
        Args:
            assignment: Binary matrix [num_tasks x num_nodes]
            
        Returns:
            Total energy
        """
        energy = 0.0
        
        # 1. Overload penalty
        energy += self._compute_overload_penalty(assignment)
        
        # 2. Communication penalty
        energy += self._compute_communication_penalty(assignment)
        
        # 3. Imbalance penalty
        energy += self._compute_imbalance_penalty(assignment)
        
        return energy
    
    def _compute_overload_penalty(self, assignment: NDArray[np.int32]) -> float:
        """
        Penalty for exceeding node capacity.
        
        For each node, penalize resource usage exceeding capacity.
        """
        penalty = 0.0
        
        for node_idx, node in enumerate(self.nodes):
            # Compute total load on this node
            node_load = {}
            
            for task_idx, task in enumerate(self.tasks):
                if assignment[task_idx, node_idx] == 1:
                    for resource, required in task.resource_requirements.items():
                        node_load[resource] = node_load.get(resource, 0.0) + required
            
            # Check each resource
            for resource, total_load in node_load.items():
                capacity = node.capacity.get(resource, 0.0)
                if total_load > capacity:
                    # Quadratic penalty for overload
                    overload = total_load - capacity
                    penalty += self.penalty_overload * (overload ** 2)
        
        return penalty
    
    def _compute_communication_penalty(self, assignment: NDArray[np.int32]) -> float:
        """
        Penalty for placing communicating tasks on different nodes.
        
        Tasks that communicate should be co-located.
        """
        penalty = 0.0
        
        for task_idx, task in enumerate(self.tasks):
            if not task.communication_partners:
                continue
            
            # Find which node this task is on
            task_node = np.argmax(assignment[task_idx])
            
            # Check communication partners
            for partner_id in task.communication_partners:
                if partner_id not in self.task_idx:
                    continue
                
                partner_idx = self.task_idx[partner_id]
                partner_node = np.argmax(assignment[partner_idx])
                
                # If on different nodes, add penalty
                if task_node != partner_node:
                    comm_cost = task.communication_cost.get(partner_id, 1.0)
                    penalty += self.penalty_communication * comm_cost
        
        return penalty
    
    def _compute_imbalance_penalty(self, assignment: NDArray[np.int32]) -> float:
        """
        Penalty for uneven load distribution.
        
        Encourage balanced load across nodes.
        """
        # Compute load per node (normalized)
        node_loads = []
        
        for node_idx, node in enumerate(self.nodes):
            total_load = 0.0
            total_capacity = sum(node.capacity.values())
            
            for task_idx, task in enumerate(self.tasks):
                if assignment[task_idx, node_idx] == 1:
                    task_load = sum(task.resource_requirements.values())
                    total_load += task_load
            
            # Normalize by capacity
            normalized_load = total_load / total_capacity if total_capacity > 0 else 0
            node_loads.append(normalized_load)
        
        # Compute variance (measure of imbalance)
        if len(node_loads) > 0:
            mean_load = np.mean(node_loads)
            variance = np.var(node_loads)
            return self.penalty_imbalance * variance
        
        return 0.0
    
    def simulated_annealing(
        self,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        iterations_per_temp: int = 100
    ) -> Tuple[NDArray[np.int32], float]:
        """
        Find optimal assignment using simulated annealing.
        
        Args:
            initial_temp: Starting temperature
            final_temp: Ending temperature
            cooling_rate: Temperature reduction factor
            iterations_per_temp: Iterations at each temperature
            
        Returns:
            (best_assignment, best_energy)
        """
        # Initialize with random valid assignment
        current = self._random_valid_assignment()
        current_energy = self.compute_energy(current)
        
        best = current.copy()
        best_energy = current_energy
        
        temp = initial_temp
        
        while temp > final_temp:
            for _ in range(iterations_per_temp):
                # Generate neighbor by moving one task
                neighbor = self._generate_neighbor(current)
                neighbor_energy = self.compute_energy(neighbor)
                
                # Accept or reject
                delta = neighbor_energy - current_energy
                
                if delta < 0:
                    # Better solution - always accept
                    current = neighbor
                    current_energy = neighbor_energy
                    
                    if current_energy < best_energy:
                        best = current.copy()
                        best_energy = current_energy
                else:
                    # Worse solution - accept with probability
                    if random.random() < math.exp(-delta / temp):
                        current = neighbor
                        current_energy = neighbor_energy
            
            # Cool down
            temp *= cooling_rate
        
        return best, best_energy
    
    def _random_valid_assignment(self) -> NDArray[np.int32]:
        """
        Generate a random valid assignment.
        
        Each task must be assigned to exactly one node.
        """
        assignment = np.zeros((self.num_tasks, self.num_nodes), dtype=np.int32)
        
        for task_idx in range(self.num_tasks):
            # Try to find a valid node
            valid_nodes = [
                node_idx for node_idx, node in enumerate(self.nodes)
                if node.can_accommodate(self.tasks[task_idx])
            ]
            
            if valid_nodes:
                chosen = random.choice(valid_nodes)
            else:
                # No valid node - assign to random one (will incur penalty)
                chosen = random.randint(0, self.num_nodes - 1)
            
            assignment[task_idx, chosen] = 1
        
        return assignment
    
    def _generate_neighbor(self, assignment: NDArray[np.int32]) -> NDArray[np.int32]:
        """
        Generate a neighbor assignment by moving one task.
        
        Args:
            assignment: Current assignment
            
        Returns:
            Neighbor assignment
        """
        neighbor = assignment.copy()
        
        # Pick random task and move it to a different node
        task_idx = random.randint(0, self.num_tasks - 1)
        current_node = np.argmax(assignment[task_idx])
        
        # Pick different node
        new_node = random.randint(0, self.num_nodes - 1)
        while new_node == current_node and self.num_nodes > 1:
            new_node = random.randint(0, self.num_nodes - 1)
        
        # Move task
        neighbor[task_idx, :] = 0
        neighbor[task_idx, new_node] = 1
        
        return neighbor
    
    def schedule(self) -> Dict[str, str]:
        """
        Run scheduler and return task assignments.
        
        Returns:
            Dictionary mapping task_id -> node_id
        """
        assignment, energy = self.simulated_annealing()
        
        # Convert to dictionary
        result = {}
        for task_idx, task in enumerate(self.tasks):
            node_idx = np.argmax(assignment[task_idx])
            result[task.task_id] = self.nodes[node_idx].node_id
        
        return result


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def _example_usage():
    """Demonstrate quantum scheduling"""
    
    print("Quantum-Inspired Scheduling (Ising Model)\n" + "="*70)
    
    # Create tasks
    tasks = [
        Task("task1", {"cpu": 2.0, "memory": 4.0}, {"task2"}, {"task2": 5.0}),
        Task("task2", {"cpu": 1.0, "memory": 2.0}, {"task1"}, {"task1": 5.0}),
        Task("task3", {"cpu": 3.0, "memory": 6.0}),
        Task("task4", {"cpu": 1.5, "memory": 3.0}),
    ]
    
    # Create nodes
    nodes = [
        Node("node1", {"cpu": 4.0, "memory": 8.0}),
        Node("node2", {"cpu": 4.0, "memory": 8.0}),
        Node("node3", {"cpu": 2.0, "memory": 4.0}),
    ]
    
    print("\n1. Tasks:")
    for task in tasks:
        print(f"   {task.task_id}: {task.resource_requirements}")
    
    print("\n2. Nodes:")
    for node in nodes:
        print(f"   {node.node_id}: {node.capacity}")
    
    # Run scheduler
    print("\n3. Running Ising scheduler...")
    scheduler = IsingScheduler(tasks, nodes)
    schedule = scheduler.schedule()
    
    print("\n4. Optimal schedule:")
    for task_id, node_id in schedule.items():
        print(f"   {task_id} -> {node_id}")
    
    # Compute final energy
    assignment = np.zeros((len(tasks), len(nodes)), dtype=np.int32)
    for i, task in enumerate(tasks):
        for j, node in enumerate(nodes):
            if schedule[task.task_id] == node.node_id:
                assignment[i, j] = 1
    
    energy = scheduler.compute_energy(assignment)
    print(f"\n5. Final energy: {energy:.2f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    _example_usage()
