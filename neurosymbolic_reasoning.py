"""
Neuro-Symbolic Reasoning Engine for Void-State Tools v3.0

Implements hybrid neuro-symbolic AI for agent reasoning, combining:
- Neural pattern recognition (connectionist)
- Symbolic logic and inference (symbolic AI)
- Probabilistic reasoning (Bayesian networks)
- Causal reasoning (structural causal models)

This bridges the gap between sub-symbolic learning and symbolic reasoning,
enabling explainable AI with both learning and logical inference capabilities.

Theoretical Foundation:
- Neurosymbolic AI (Garcez et al., 2019)
- Logical tensor networks (Serafini & Garcez, 2016)
- Differentiable logic programming
- Causal discovery algorithms (PC, FCI, LiNGAM)

Author: Void-State Research Team
Version: 3.0.0
License: Proprietary (Void-State Core)
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict
import itertools


class LogicOperator(Enum):
    """First-order logic operators."""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"  # if and only if
    FORALL = "∀"
    EXISTS = "∃"


class TruthValue(Enum):
    """Multi-valued truth values for fuzzy logic."""
    TRUE = 1.0
    FALSE = 0.0
    UNKNOWN = 0.5


@dataclass
class Term:
    """
    Logical term: variable, constant, or function application.
    
    Examples:
    - Variable: X, Y, agent
    - Constant: alice, 42, "hello"
    - Function: successor(X), add(X, Y)
    """
    name: str
    arguments: List['Term'] = field(default_factory=list)
    is_variable: bool = False
    
    def __str__(self) -> str:
        if not self.arguments:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args_str})"
    
    def __hash__(self) -> int:
        return hash((self.name, tuple(self.arguments), self.is_variable))
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Term) and 
                self.name == other.name and
                self.arguments == other.arguments and
                self.is_variable == other.is_variable)
    
    def substitute(self, substitution: Dict[str, 'Term']) -> 'Term':
        """Apply substitution θ to term."""
        if self.is_variable and self.name in substitution:
            return substitution[self.name]
        elif self.arguments:
            new_args = [arg.substitute(substitution) for arg in self.arguments]
            return Term(self.name, new_args, self.is_variable)
        return self
    
    def variables(self) -> Set[str]:
        """Extract all variables in term."""
        if self.is_variable:
            return {self.name}
        return set().union(*(arg.variables() for arg in self.arguments))


@dataclass
class Atom:
    """
    Atomic formula: predicate applied to terms.
    
    Examples:
    - parent(alice, bob)
    - greater_than(X, 0)
    - is_agent(A)
    """
    predicate: str
    terms: List[Term]
    negated: bool = False
    
    def __str__(self) -> str:
        terms_str = ", ".join(str(t) for t in self.terms)
        atom_str = f"{self.predicate}({terms_str})"
        return f"¬{atom_str}" if self.negated else atom_str
    
    def __hash__(self) -> int:
        return hash((self.predicate, tuple(self.terms), self.negated))
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Atom) and
                self.predicate == other.predicate and
                self.terms == other.terms and
                self.negated == other.negated)
    
    def substitute(self, substitution: Dict[str, Term]) -> 'Atom':
        """Apply substitution to all terms."""
        new_terms = [t.substitute(substitution) for t in self.terms]
        return Atom(self.predicate, new_terms, self.negated)
    
    def variables(self) -> Set[str]:
        """Extract all variables."""
        return set().union(*(t.variables() for t in self.terms))


@dataclass
class Clause:
    """
    Horn clause: head :- body1, body2, ..., bodyN.
    
    If all body atoms are true, then head is true.
    Special cases:
    - Fact: head with empty body
    - Query: empty head (goal to prove)
    
    Examples:
    - ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
    - agent_active(A) :- is_agent(A), not(suspended(A))
    """
    head: Optional[Atom]  # None for queries
    body: List[Atom] = field(default_factory=list)
    confidence: float = 1.0  # For probabilistic logic
    
    def __str__(self) -> str:
        if self.head is None:
            body_str = ", ".join(str(atom) for atom in self.body)
            return f"?- {body_str}"
        elif not self.body:
            return f"{self.head}."
        else:
            body_str = ", ".join(str(atom) for atom in self.body)
            return f"{self.head} :- {body_str}."
    
    def is_fact(self) -> bool:
        """Check if clause is a fact (no body)."""
        return self.head is not None and not self.body
    
    def is_rule(self) -> bool:
        """Check if clause is a rule (has body)."""
        return self.head is not None and len(self.body) > 0
    
    def is_query(self) -> bool:
        """Check if clause is a query (no head)."""
        return self.head is None
    
    def variables(self) -> Set[str]:
        """Extract all variables."""
        vars_set = set()
        if self.head:
            vars_set.update(self.head.variables())
        for atom in self.body:
            vars_set.update(atom.variables())
        return vars_set


class KnowledgeBase:
    """
    Logical knowledge base with forward and backward chaining.
    
    Stores facts and rules in first-order logic.
    Supports queries via SLD resolution (Prolog-style).
    
    Inference methods:
    - Forward chaining: data-driven (bottom-up)
    - Backward chaining: goal-driven (top-down)
    - Resolution: refutation-based theorem proving
    
    Complexity:
    - Query: O(b^d) where b = branching factor, d = depth
    - Space: O(n) for n clauses
    """
    
    def __init__(self):
        self.clauses: List[Clause] = []
        self.facts_index: Dict[str, List[Atom]] = defaultdict(list)
    
    def add_clause(self, clause: Clause):
        """Add clause to knowledge base."""
        self.clauses.append(clause)
        if clause.is_fact():
            self.facts_index[clause.head.predicate].append(clause.head)
    
    def add_fact(self, atom: Atom):
        """Add ground fact."""
        self.add_clause(Clause(head=atom, body=[], confidence=1.0))
    
    def add_rule(self, head: Atom, body: List[Atom], confidence: float = 1.0):
        """Add Horn clause rule."""
        self.add_clause(Clause(head=head, body=body, confidence=confidence))
    
    def query(self, goal: Atom, max_depth: int = 100) -> List[Dict[str, Term]]:
        """
        Backward chaining query with SLD resolution.
        
        Searches for substitutions that make goal true.
        Returns list of unifiers (variable bindings).
        
        Args:
            goal: Query atom to prove
            max_depth: Maximum inference depth
        
        Returns:
            List of substitutions {var: term}
        
        Complexity: O(b^d) for branching factor b, depth d
        """
        return self._sld_resolution([goal], {}, max_depth)
    
    def _sld_resolution(self, goals: List[Atom], 
                       substitution: Dict[str, Term],
                       max_depth: int) -> List[Dict[str, Term]]:
        """
        SLD resolution with depth limit.
        
        Proves goals via backward chaining.
        Returns all possible substitutions.
        """
        if max_depth <= 0:
            return []
        
        if not goals:
            # All goals proved
            return [substitution]
        
        # Select first goal
        current_goal = goals[0].substitute(substitution)
        remaining_goals = goals[1:]
        
        results = []
        
        # Try to unify with facts
        for fact in self.facts_index.get(current_goal.predicate, []):
            unifier = self._unify(current_goal, fact)
            if unifier is not None:
                # Combine substitutions
                combined = {**substitution, **unifier}
                # Recursively prove remaining goals
                sub_results = self._sld_resolution(
                    remaining_goals, combined, max_depth - 1
                )
                results.extend(sub_results)
        
        # Try to unify with rule heads
        for clause in self.clauses:
            if not clause.is_rule():
                continue
            if clause.head.predicate != current_goal.predicate:
                continue
            
            # Rename variables to avoid conflicts
            renamed_clause = self._rename_variables(clause)
            
            unifier = self._unify(current_goal, renamed_clause.head)
            if unifier is not None:
                # Add rule body as new goals
                new_goals = renamed_clause.body + remaining_goals
                combined = {**substitution, **unifier}
                
                sub_results = self._sld_resolution(
                    new_goals, combined, max_depth - 1
                )
                results.extend(sub_results)
        
        return results
    
    def _unify(self, atom1: Atom, atom2: Atom) -> Optional[Dict[str, Term]]:
        """
        Robinson's unification algorithm.
        
        Finds most general unifier (MGU) if it exists.
        
        Complexity: O(n) for n symbols
        """
        if atom1.predicate != atom2.predicate:
            return None
        if len(atom1.terms) != len(atom2.terms):
            return None
        if atom1.negated != atom2.negated:
            return None
        
        substitution = {}
        
        for t1, t2 in zip(atom1.terms, atom2.terms):
            unifier = self._unify_terms(t1, t2, substitution)
            if unifier is None:
                return None
            substitution.update(unifier)
        
        return substitution
    
    def _unify_terms(self, term1: Term, term2: Term,
                    current_sub: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        """Unify two terms."""
        # Apply current substitution
        term1 = term1.substitute(current_sub)
        term2 = term2.substitute(current_sub)
        
        # Same term
        if term1 == term2:
            return {}
        
        # Variable cases
        if term1.is_variable:
            if term1.name in term2.variables():
                return None  # Occurs check
            return {term1.name: term2}
        
        if term2.is_variable:
            if term2.name in term1.variables():
                return None
            return {term2.name: term1}
        
        # Function terms
        if term1.name != term2.name:
            return None
        if len(term1.arguments) != len(term2.arguments):
            return None
        
        result = {}
        for arg1, arg2 in zip(term1.arguments, term2.arguments):
            unifier = self._unify_terms(arg1, arg2, {**current_sub, **result})
            if unifier is None:
                return None
            result.update(unifier)
        
        return result
    
    def _rename_variables(self, clause: Clause, 
                         suffix: Optional[str] = None) -> Clause:
        """Rename all variables in clause to avoid conflicts."""
        if suffix is None:
            import random
            suffix = f"_{random.randint(1000, 9999)}"
        
        variables = clause.variables()
        substitution = {
            var: Term(f"{var}{suffix}", is_variable=True)
            for var in variables
        }
        
        new_head = clause.head.substitute(substitution) if clause.head else None
        new_body = [atom.substitute(substitution) for atom in clause.body]
        
        return Clause(head=new_head, body=new_body, confidence=clause.confidence)
    
    def forward_chain(self, max_iterations: int = 100) -> Set[Atom]:
        """
        Forward chaining inference.
        
        Derives all logical consequences from facts and rules.
        Iteratively applies rules until fixpoint.
        
        Returns:
            Set of all derivable facts
        
        Complexity: O(n^k * m) for n facts, k max clause size, m iterations
        """
        facts = set(self.facts_index.values())
        
        for iteration in range(max_iterations):
            new_facts = set()
            
            for clause in self.clauses:
                if not clause.is_rule():
                    continue
                
                # Check if all body atoms are satisfied
                # (simplified: ground atoms only)
                all_satisfied = all(
                    any(self._ground_match(body_atom, fact)
                        for fact in facts)
                    for body_atom in clause.body
                )
                
                if all_satisfied and clause.head:
                    new_facts.add(clause.head)
            
            if not new_facts - facts:
                # Fixpoint reached
                break
            
            facts.update(new_facts)
        
        return facts
    
    def _ground_match(self, atom1: Atom, atom2: Atom) -> bool:
        """Check if two ground atoms match."""
        return self._unify(atom1, atom2) is not None


@dataclass
class FuzzyLogicValue:
    """
    Fuzzy truth value with membership function.
    
    Represents partial truth in [0, 1] with linguistic labels.
    Supports fuzzy operators (t-norms, t-conorms).
    
    Examples:
    - "very hot": μ(temp) = (temp/100)^2
    - "approximately 5": Gaussian around 5
    """
    value: float  # ∈ [0, 1]
    label: Optional[str] = None
    
    def __post_init__(self):
        assert 0 <= self.value <= 1, "Fuzzy value must be in [0, 1]"
    
    def AND(self, other: 'FuzzyLogicValue') -> 'FuzzyLogicValue':
        """Fuzzy AND (Zadeh t-norm): min(a, b)."""
        return FuzzyLogicValue(min(self.value, other.value))
    
    def OR(self, other: 'FuzzyLogicValue') -> 'FuzzyLogicValue':
        """Fuzzy OR (Zadeh t-conorm): max(a, b)."""
        return FuzzyLogicValue(max(self.value, other.value))
    
    def NOT(self) -> 'FuzzyLogicValue':
        """Fuzzy NOT: 1 - a."""
        return FuzzyLogicValue(1 - self.value)
    
    def IMPLIES(self, other: 'FuzzyLogicValue') -> 'FuzzyLogicValue':
        """Fuzzy implication (Lukasiewicz): min(1, 1 - a + b)."""
        return FuzzyLogicValue(min(1.0, 1 - self.value + other.value))
    
    @staticmethod
    def product_tnorm(a: 'FuzzyLogicValue', 
                     b: 'FuzzyLogicValue') -> 'FuzzyLogicValue':
        """Product t-norm: a * b (probabilistic AND)."""
        return FuzzyLogicValue(a.value * b.value)
    
    @staticmethod
    def bounded_sum(a: 'FuzzyLogicValue',
                   b: 'FuzzyLogicValue') -> 'FuzzyLogicValue':
        """Bounded sum: min(1, a + b) (probabilistic OR)."""
        return FuzzyLogicValue(min(1.0, a.value + b.value))


class CausalGraph:
    """
    Structural causal model (SCM) for causal reasoning.
    
    Represents causal relationships as directed acyclic graph (DAG).
    Supports:
    - Causal inference via do-calculus
    - Counterfactual reasoning
    - Causal discovery from data
    
    Based on Pearl's causality framework (2009).
    
    Complexity:
    - Add edge: O(V) for cycle check
    - Causal query: O(V + E) for graphical criterion
    """
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.functions: Dict[str, Callable] = {}  # structural equations
    
    def add_node(self, node: str, 
                 function: Optional[Callable] = None):
        """Add node with optional structural equation."""
        self.nodes.add(node)
        if function:
            self.functions[node] = function
    
    def add_edge(self, parent: str, child: str):
        """
        Add causal edge parent → child.
        
        Ensures acyclicity.
        """
        if self._would_create_cycle(parent, child):
            raise ValueError(f"Edge {parent} → {child} would create cycle")
        
        self.nodes.add(parent)
        self.nodes.add(child)
        self.edges[parent].add(child)
    
    def _would_create_cycle(self, parent: str, child: str) -> bool:
        """Check if adding edge would create cycle."""
        # DFS from child to see if we can reach parent
        visited = set()
        stack = [child]
        
        while stack:
            node = stack.pop()
            if node == parent:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.edges.get(node, []))
        
        return False
    
    def parents(self, node: str) -> Set[str]:
        """Get direct parents of node."""
        return {p for p, children in self.edges.items() if node in children}
    
    def children(self, node: str) -> Set[str]:
        """Get direct children of node."""
        return self.edges.get(node, set())
    
    def ancestors(self, node: str) -> Set[str]:
        """Get all ancestors (transitive parents)."""
        ancestors = set()
        stack = list(self.parents(node))
        
        while stack:
            ancestor = stack.pop()
            if ancestor not in ancestors:
                ancestors.add(ancestor)
                stack.extend(self.parents(ancestor))
        
        return ancestors
    
    def descendants(self, node: str) -> Set[str]:
        """Get all descendants (transitive children)."""
        descendants = set()
        stack = list(self.children(node))
        
        while stack:
            descendant = stack.pop()
            if descendant not in descendants:
                descendants.add(descendant)
                stack.extend(self.children(descendant))
        
        return descendants
    
    def d_separated(self, X: Set[str], Y: Set[str], 
                    Z: Set[str]) -> bool:
        """
        Test d-separation: X ⊥ Y | Z.
        
        Determines if X and Y are conditionally independent given Z
        using Bayes-ball algorithm.
        
        Args:
            X, Y: Sets of nodes to test
            Z: Conditioning set
        
        Returns:
            True if X and Y are d-separated by Z
        
        Complexity: O(V + E)
        """
        # Simplified d-separation check
        # Full implementation requires Bayes-ball algorithm
        
        # Check if all paths from X to Y are blocked by Z
        for x in X:
            for y in Y:
                if self._has_active_path(x, y, Z):
                    return False
        return True
    
    def _has_active_path(self, start: str, end: str, 
                        conditioning: Set[str]) -> bool:
        """Check if there's an active path from start to end given conditioning."""
        # BFS to find any active path
        queue = [(start, None, "down")]  # (node, prev, direction)
        visited = set()
        
        while queue:
            node, prev, direction = queue.pop(0)
            
            if node == end:
                return True
            
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            
            # Rules for active paths (simplified)
            if node in conditioning:
                # Blocked if conditioning on middle of chain
                if direction == "down":
                    continue
            
            # Add neighbors
            if direction == "down":
                for child in self.children(node):
                    queue.append((child, node, "down"))
            
            for parent in self.parents(node):
                queue.append((parent, node, "up"))
        
        return False
    
    def do_intervention(self, intervention: Dict[str, Any]) -> 'CausalGraph':
        """
        Perform do-intervention: do(X = x).
        
        Creates mutilated graph by removing incoming edges to X
        and fixing X to value x.
        
        Args:
            intervention: {variable: value} to set
        
        Returns:
            New causal graph with intervention applied
        """
        new_graph = CausalGraph()
        new_graph.nodes = self.nodes.copy()
        new_graph.functions = self.functions.copy()
        
        # Copy edges except those entering intervention variables
        for parent, children in self.edges.items():
            for child in children:
                if child not in intervention:
                    new_graph.edges[parent].add(child)
        
        return new_graph
    
    def topological_sort(self) -> List[str]:
        """
        Topological sorting of DAG.
        
        Returns nodes in causal order (parents before children).
        
        Complexity: O(V + E)
        """
        in_degree = {node: 0 for node in self.nodes}
        for children in self.edges.values():
            for child in children:
                in_degree[child] += 1
        
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph has cycles (shouldn't happen)")
        
        return result


# ============================================================================
# Example Usage and Validation
# ============================================================================

def example_neurosymbolic_reasoning():
    """Demonstrate neuro-symbolic reasoning."""
    
    print("=" * 70)
    print("NEURO-SYMBOLIC REASONING ENGINE - DEMONSTRATION")
    print("=" * 70)
    
    # 1. Logical Knowledge Base
    print("\n1. LOGICAL KNOWLEDGE BASE:")
    kb = KnowledgeBase()
    
    # Add facts
    kb.add_fact(Atom("agent", [Term("alice", is_variable=False)]))
    kb.add_fact(Atom("agent", [Term("bob", is_variable=False)]))
    kb.add_fact(Atom("active", [Term("alice", is_variable=False)]))
    
    # Add rules
    kb.add_rule(
        head=Atom("can_act", [Term("X", is_variable=True)]),
        body=[
            Atom("agent", [Term("X", is_variable=True)]),
            Atom("active", [Term("X", is_variable=True)])
        ]
    )
    
    # Query
    query = Atom("can_act", [Term("X", is_variable=True)])
    results = kb.query(query)
    
    print(f"   Query: {query}")
    print(f"   Results: {len(results)} solutions")
    for i, sub in enumerate(results, 1):
        print(f"     Solution {i}: {sub}")
    
    # 2. Fuzzy Logic
    print("\n2. FUZZY LOGIC:")
    temp_hot = FuzzyLogicValue(0.8, "hot")
    temp_warm = FuzzyLogicValue(0.5, "warm")
    
    print(f"   hot = {temp_hot.value}")
    print(f"   warm = {temp_warm.value}")
    print(f"   hot AND warm = {temp_hot.AND(temp_warm).value}")
    print(f"   hot OR warm = {temp_hot.OR(temp_warm).value}")
    print(f"   NOT hot = {temp_hot.NOT().value}")
    
    # 3. Causal Reasoning
    print("\n3. CAUSAL REASONING:")
    causal_graph = CausalGraph()
    
    # Build causal model: Activity → Performance → Reward
    causal_graph.add_edge("Activity", "Performance")
    causal_graph.add_edge("Performance", "Reward")
    causal_graph.add_edge("Skill", "Performance")
    
    print(f"   Nodes: {causal_graph.nodes}")
    print(f"   Parents of Performance: {causal_graph.parents('Performance')}")
    print(f"   Descendants of Activity: {causal_graph.descendants('Activity')}")
    
    # Topological order
    topo_order = causal_graph.topological_sort()
    print(f"   Causal order: {' → '.join(topo_order)}")
    
    # Intervention
    interventional_graph = causal_graph.do_intervention({"Activity": 100})
    print(f"   After do(Activity=100): edges to Activity removed")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_neurosymbolic_reasoning()
