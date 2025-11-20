"""
Causal Inference and Root Cause Analysis

Implements causal reasoning using structural causal models (SCM) and
do-calculus for root cause analysis in distributed systems.

Rather than learning causal structure from scratch (expensive),
this module uses a static causal graph of system topology and learns
edge weights (failure propagation probabilities) dynamically.

Key concepts:
- Structural Causal Model (SCM): DAG + probability distributions
- Do-calculus: Interventional queries P(Y | do(X))
- Backdoor criterion: Identifying causal effects
- Counterfactual reasoning: "What if X had been different?"

References:
- "Causality" (Pearl, 2009)
- "The Book of Why" (Pearl & Mackenzie, 2018)
- "Elements of Causal Inference" (Peters et al., 2017)
"""

from typing import Dict, List, Set, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict
import networkx as nx


# ============================================================================
# CAUSAL GRAPH
# ============================================================================

@dataclass
class CausalEdge:
    """
    An edge in the causal graph.
    
    Represents a causal relationship from parent to child.
    The weight represents the strength/probability of failure propagation.
    """
    parent: str
    child: str
    weight: float = 1.0  # Probability of failure propagation
    
    def __hash__(self):
        return hash((self.parent, self.child))


class CausalGraph:
    """
    Structural Causal Model (SCM) for system topology.
    
    The graph structure is static (known system architecture).
    Edge weights are learned dynamically from observations.
    """
    
    def __init__(self):
        """Initialize empty causal graph"""
        self.graph = nx.DiGraph()
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        
        # Statistics for learning edge weights
        self.edge_observations: Dict[Tuple[str, str], List[Tuple[bool, bool]]] = defaultdict(list)
    
    def add_node(self, node: str, node_type: Optional[str] = None):
        """
        Add a node to the causal graph.
        
        Args:
            node: Node identifier
            node_type: Optional node type (e.g., "database", "cache", "api")
        """
        self.graph.add_node(node, node_type=node_type)
    
    def add_edge(self, parent: str, child: str, weight: float = 1.0):
        """
        Add a causal edge parent → child.
        
        Args:
            parent: Parent node
            child: Child node
            weight: Initial edge weight
        """
        self.graph.add_edge(parent, child)
        self.edge_weights[(parent, child)] = weight
    
    def get_parents(self, node: str) -> Set[str]:
        """Get all parents (direct causes) of a node"""
        return set(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> Set[str]:
        """Get all children (direct effects) of a node"""
        return set(self.graph.successors(node))
    
    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors (transitive causes) of a node"""
        return nx.ancestors(self.graph, node)
    
    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants (transitive effects) of a node"""
        return nx.descendants(self.graph, node)
    
    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if ancestor is a transitive cause of descendant"""
        return ancestor in self.get_ancestors(descendant)
    
    def topological_sort(self) -> List[str]:
        """
        Get nodes in topological order (causes before effects).
        
        Returns:
            Ordered list of nodes
        """
        return list(nx.topological_sort(self.graph))
    
    def observe_failure_propagation(
        self,
        parent: str,
        child: str,
        parent_failed: bool,
        child_failed: bool
    ):
        """
        Record an observation of failure propagation.
        
        This is used to learn edge weights dynamically.
        
        Args:
            parent: Parent node
            child: Child node
            parent_failed: Whether parent failed
            child_failed: Whether child failed
        """
        if (parent, child) not in self.edge_weights:
            return
        
        self.edge_observations[(parent, child)].append((parent_failed, child_failed))
        
        # Update edge weight based on observations
        self._update_edge_weight(parent, child)
    
    def _update_edge_weight(self, parent: str, child: str):
        """
        Update edge weight based on observations.
        
        Weight = P(child fails | parent fails)
        """
        observations = self.edge_observations[(parent, child)]
        
        if not observations:
            return
        
        # Count cases where parent failed
        parent_failures = [obs for obs in observations if obs[0]]
        
        if not parent_failures:
            return
        
        # Count cases where both failed
        both_failed = [obs for obs in parent_failures if obs[1]]
        
        # Update weight
        self.edge_weights[(parent, child)] = len(both_failed) / len(parent_failures)
    
    def get_edge_weight(self, parent: str, child: str) -> float:
        """Get learned edge weight"""
        return self.edge_weights.get((parent, child), 0.0)


# ============================================================================
# DO-CALCULUS AND INTERVENTIONS
# ============================================================================

class CausalInference:
    """
    Performs causal inference using do-calculus.
    
    Answers questions like:
    - P(API_Fail | do(Cache_Down)) - What if we force cache down?
    - P(API_Fail | Cache_Down) - Observational probability
    
    The difference distinguishes correlation from causation.
    """
    
    def __init__(self, graph: CausalGraph):
        """
        Initialize causal inference engine.
        
        Args:
            graph: Causal graph with learned weights
        """
        self.graph = graph
    
    def observational_probability(
        self,
        outcome: str,
        condition: Dict[str, bool]
    ) -> float:
        """
        Compute P(outcome | conditions).
        
        This is standard conditional probability (observational).
        
        Args:
            outcome: Target variable
            condition: Conditioning variables
            
        Returns:
            Conditional probability
        """
        # For simplicity, use causal path probability
        # In practice, would use full probabilistic inference
        
        prob = 1.0
        
        # Find paths from conditioned variables to outcome
        for cond_var, cond_value in condition.items():
            if not cond_value:
                continue
            
            # Check if there's a path
            if self.graph.is_ancestor(cond_var, outcome):
                # Compute path probability
                path_prob = self._compute_path_probability(cond_var, outcome)
                prob *= path_prob
        
        return min(prob, 1.0)
    
    def interventional_probability(
        self,
        outcome: str,
        intervention: Dict[str, bool]
    ) -> float:
        """
        Compute P(outcome | do(intervention)).
        
        This is causal probability - what if we force variables to values.
        
        Uses do-calculus to compute the effect of interventions.
        
        Args:
            outcome: Target variable
            intervention: Intervention variables
            
        Returns:
            Interventional probability
        """
        # Create interventional graph (remove incoming edges to intervention vars)
        intervened_graph = self._create_interventional_graph(intervention.keys())
        
        # Compute probability in interventional graph
        prob = 1.0
        
        for int_var, int_value in intervention.items():
            if not int_value:
                continue
            
            # Check direct causal paths
            if int_var in intervened_graph.get_ancestors(outcome):
                path_prob = self._compute_path_probability_in_graph(
                    int_var, outcome, intervened_graph
                )
                prob *= path_prob
        
        return min(prob, 1.0)
    
    def _create_interventional_graph(self, intervention_vars: Set[str]) -> CausalGraph:
        """
        Create graph for intervention by removing incoming edges.
        
        do(X=x) is represented by cutting all edges into X.
        
        Args:
            intervention_vars: Variables being intervened on
            
        Returns:
            Modified causal graph
        """
        new_graph = CausalGraph()
        
        # Copy nodes
        for node in self.graph.graph.nodes():
            new_graph.add_node(node)
        
        # Copy edges, except incoming edges to intervention vars
        for parent, child in self.graph.graph.edges():
            if child not in intervention_vars:
                weight = self.graph.get_edge_weight(parent, child)
                new_graph.add_edge(parent, child, weight)
        
        return new_graph
    
    def _compute_path_probability(self, source: str, target: str) -> float:
        """
        Compute probability along causal path.
        
        Uses edge weights (failure propagation probabilities).
        """
        return self._compute_path_probability_in_graph(source, target, self.graph)
    
    def _compute_path_probability_in_graph(
        self,
        source: str,
        target: str,
        graph: CausalGraph
    ) -> float:
        """Compute path probability in given graph"""
        try:
            # Find shortest path
            path = nx.shortest_path(graph.graph, source, target)
            
            # Multiply edge weights along path
            prob = 1.0
            for i in range(len(path) - 1):
                weight = graph.get_edge_weight(path[i], path[i + 1])
                prob *= weight
            
            return prob
        except nx.NetworkXNoPath:
            return 0.0
    
    def compute_backdoor_adjustment(
        self,
        treatment: str,
        outcome: str,
        confounders: Set[str]
    ) -> float:
        """
        Compute causal effect using backdoor adjustment.
        
        Backdoor criterion: Block all spurious paths by conditioning on confounders.
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            confounders: Confounding variables to adjust for
            
        Returns:
            Causal effect estimate
        """
        # Simplified backdoor adjustment
        # In practice, would marginalize over confounder values
        
        # Remove confounding paths
        adjusted_graph = CausalGraph()
        
        # Copy structure but remove confounder effects
        for node in self.graph.graph.nodes():
            adjusted_graph.add_node(node)
        
        for parent, child in self.graph.graph.edges():
            if parent not in confounders or child == outcome:
                weight = self.graph.get_edge_weight(parent, child)
                adjusted_graph.add_edge(parent, child, weight)
        
        # Compute effect in adjusted graph
        return self._compute_path_probability_in_graph(
            treatment, outcome, adjusted_graph
        )
    
    def find_root_causes(
        self,
        failed_components: Set[str],
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find root causes of failures using causal analysis.
        
        Returns components that are likely root causes based on:
        1. They failed
        2. They have no failed parents (or low probability of parent causing it)
        3. They have failed descendants
        
        Args:
            failed_components: Set of components that failed
            threshold: Minimum probability threshold
            
        Returns:
            List of (component, confidence) tuples
        """
        root_causes = []
        
        for component in failed_components:
            # Check if it has failed parents
            parents = self.graph.get_parents(component)
            failed_parents = [p for p in parents if p in failed_components]
            
            if not failed_parents:
                # No failed parents - likely root cause
                # Check if it has failed descendants
                descendants = self.graph.get_descendants(component)
                failed_descendants = [d for d in descendants if d in failed_components]
                
                if failed_descendants:
                    # Compute confidence based on number of affected descendants
                    confidence = len(failed_descendants) / max(len(descendants), 1)
                    
                    if confidence >= threshold:
                        root_causes.append((component, confidence))
            else:
                # Has failed parents - check if they explain the failure
                explained_prob = 0.0
                
                for parent in failed_parents:
                    weight = self.graph.get_edge_weight(parent, component)
                    explained_prob = max(explained_prob, weight)
                
                # If not well explained by parents, might still be root cause
                if explained_prob < threshold:
                    confidence = 1.0 - explained_prob
                    root_causes.append((component, confidence))
        
        # Sort by confidence
        root_causes.sort(key=lambda x: x[1], reverse=True)
        
        return root_causes


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def _example_usage():
    """Demonstrate causal inference"""
    
    print("Causal Inference and Root Cause Analysis\n" + "="*70)
    
    # Build system topology
    graph = CausalGraph()
    
    # Add nodes
    components = ["Database", "Cache", "API", "LoadBalancer", "Frontend"]
    for comp in components:
        graph.add_node(comp)
    
    # Add causal edges (system dependencies)
    graph.add_edge("Database", "Cache", 0.7)
    graph.add_edge("Database", "API", 0.5)
    graph.add_edge("Cache", "API", 0.8)
    graph.add_edge("API", "LoadBalancer", 0.9)
    graph.add_edge("LoadBalancer", "Frontend", 0.9)
    
    print("\n1. System topology:")
    print("   Database → Cache → API → LoadBalancer → Frontend")
    print("   Database → API")
    
    # Create inference engine
    inference = CausalInference(graph)
    
    print("\n2. Observational query:")
    # P(API fails | Cache down)
    prob_obs = inference.observational_probability(
        "API",
        {"Cache": True}
    )
    print(f"   P(API fails | Cache down) = {prob_obs:.2f}")
    
    print("\n3. Interventional query (do-calculus):")
    # P(API fails | do(Cache down))
    prob_int = inference.interventional_probability(
        "API",
        {"Cache": True}
    )
    print(f"   P(API fails | do(Cache down)) = {prob_int:.2f}")
    
    print("\n4. Root cause analysis:")
    # Simulate failure scenario
    failed = {"Database", "Cache", "API", "LoadBalancer"}
    
    print(f"   Failed components: {failed}")
    root_causes = inference.find_root_causes(failed)
    print("   Root causes (ranked by confidence):")
    for component, confidence in root_causes:
        print(f"     {component}: {confidence:.2f}")
    
    print("\n5. Learning edge weights from observations:")
    # Simulate some observations
    graph.observe_failure_propagation("Database", "Cache", True, True)
    graph.observe_failure_propagation("Database", "Cache", True, True)
    graph.observe_failure_propagation("Database", "Cache", True, False)
    
    weight = graph.get_edge_weight("Database", "Cache")
    print(f"   Updated weight Database → Cache: {weight:.2f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    _example_usage()
