"""
Advanced Type System for Void-State Tools

This module provides a mathematically rigorous type system with category-theoretic
foundations, enabling formal reasoning about tool behavior and composition.

References:
- Category Theory for the Sciences (Spivak, 2014)
- Type Theory and Formal Proof (Nederpelt & Geuvers, 2014)
- Homotopy Type Theory (Univalent Foundations, 2013)
"""

from typing import (
    TypeVar, Generic, Callable, Protocol, runtime_checkable,
    Any, Optional, Union, Tuple, List, Dict, Set, FrozenSet
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# CATEGORY THEORY FOUNDATIONS
# ============================================================================

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


@runtime_checkable
class Morphism(Protocol[T, U]):
    """A morphism (arrow) in a category - structure-preserving map between objects"""
    
    def __call__(self, source: T) -> U:
        """Apply the morphism"""
        ...
    
    def compose(self, other: 'Morphism[U, V]') -> 'Morphism[T, V]':
        """Compose with another morphism (categorical composition)"""
        ...


class Identity(Morphism[T, T]):
    """Identity morphism - maps object to itself"""
    
    def __call__(self, x: T) -> T:
        return x
    
    def compose(self, other: Morphism[T, U]) -> Morphism[T, U]:
        return other


@dataclass(frozen=True)
class ComposedMorphism(Morphism[T, V]):
    """Composition of two morphisms: h = g ∘ f"""
    f: Morphism[T, U]
    g: Morphism[U, V]
    
    def __call__(self, x: T) -> V:
        return self.g(self.f(x))
    
    def compose(self, other: Morphism[V, Any]) -> Morphism[T, Any]:
        return ComposedMorphism(self, other)


# ============================================================================
# ALGEBRAIC DATA TYPES
# ============================================================================

class ADT(ABC):
    """Base class for Algebraic Data Types"""
    pass


@dataclass(frozen=True)
class Sum(ADT, Generic[T, U]):
    """Sum type (coproduct): Either T or U"""
    value: Union[T, U]
    is_left: bool
    
    @staticmethod
    def left(value: T) -> 'Sum[T, U]':
        return Sum(value, True)
    
    @staticmethod
    def right(value: U) -> 'Sum[T, U]':
        return Sum(value, False)
    
    def match(self, left_fn: Callable[[T], V], right_fn: Callable[[U], V]) -> V:
        """Pattern matching on sum type"""
        if self.is_left:
            return left_fn(self.value)  # type: ignore
        else:
            return right_fn(self.value)  # type: ignore


@dataclass(frozen=True)
class Product(ADT, Generic[T, U]):
    """Product type (pair): T × U"""
    fst: T
    snd: U
    
    def map_fst(self, f: Callable[[T], V]) -> 'Product[V, U]':
        return Product(f(self.fst), self.snd)
    
    def map_snd(self, f: Callable[[U], V]) -> 'Product[T, V]':
        return Product(self.fst, f(self.snd))


@dataclass(frozen=True)
class Maybe(ADT, Generic[T]):
    """Option type: represents nullable values type-safely"""
    value: Optional[T]
    
    @staticmethod
    def some(value: T) -> 'Maybe[T]':
        return Maybe(value)
    
    @staticmethod
    def none() -> 'Maybe[T]':
        return Maybe(None)
    
    def is_some(self) -> bool:
        return self.value is not None
    
    def is_none(self) -> bool:
        return self.value is None
    
    def map(self, f: Callable[[T], U]) -> 'Maybe[U]':
        if self.is_some():
            return Maybe.some(f(self.value))  # type: ignore
        return Maybe.none()
    
    def flat_map(self, f: Callable[[T], 'Maybe[U]']) -> 'Maybe[U]':
        if self.is_some():
            return f(self.value)  # type: ignore
        return Maybe.none()
    
    def get_or_else(self, default: T) -> T:
        return self.value if self.is_some() else default


# ============================================================================
# TEMPORAL TYPES (WITH METRIC SPACE STRUCTURE)
# ============================================================================

@dataclass(frozen=True)
class Timestamp:
    """
    Precise timestamp with nanosecond resolution.
    Forms a totally ordered set with metric d(t1, t2) = |t1 - t2|
    """
    seconds: int
    nanoseconds: int
    
    def __post_init__(self):
        assert 0 <= self.nanoseconds < 1_000_000_000
    
    def to_float(self) -> float:
        """Convert to floating point seconds"""
        return self.seconds + self.nanoseconds / 1e9
    
    def __sub__(self, other: 'Timestamp') -> 'Duration':
        """Temporal difference"""
        total_ns = (self.seconds - other.seconds) * 1_000_000_000
        total_ns += self.nanoseconds - other.nanoseconds
        return Duration(nanoseconds=total_ns)
    
    def __add__(self, duration: 'Duration') -> 'Timestamp':
        """Temporal addition"""
        total_ns = self.nanoseconds + duration.nanoseconds
        seconds = self.seconds + duration.seconds + (total_ns // 1_000_000_000)
        nanoseconds = total_ns % 1_000_000_000
        return Timestamp(seconds, nanoseconds)
    
    def __lt__(self, other: 'Timestamp') -> bool:
        return (self.seconds, self.nanoseconds) < (other.seconds, other.nanoseconds)


@dataclass(frozen=True)
class Duration:
    """
    Time duration with nanosecond precision.
    Forms an additive monoid: (Duration, +, Duration(0))
    """
    seconds: int = 0
    nanoseconds: int = 0
    
    def __post_init__(self):
        # Normalize
        object.__setattr__(self, 'seconds', self.seconds + self.nanoseconds // 1_000_000_000)
        object.__setattr__(self, 'nanoseconds', self.nanoseconds % 1_000_000_000)
    
    def to_nanoseconds(self) -> int:
        return self.seconds * 1_000_000_000 + self.nanoseconds
    
    def __add__(self, other: 'Duration') -> 'Duration':
        return Duration(
            self.seconds + other.seconds,
            self.nanoseconds + other.nanoseconds
        )
    
    def __mul__(self, scalar: float) -> 'Duration':
        total_ns = int((self.seconds * 1_000_000_000 + self.nanoseconds) * scalar)
        return Duration(nanoseconds=total_ns)


@dataclass(frozen=True)
class TimeInterval:
    """
    Time interval [start, end).
    Forms a partially ordered set under the "precedes" relation.
    """
    start: Timestamp
    end: Timestamp
    
    def __post_init__(self):
        assert self.start < self.end, "Start must precede end"
    
    def duration(self) -> Duration:
        return self.end - self.start
    
    def contains(self, t: Timestamp) -> bool:
        """Check if timestamp is in interval [start, end)"""
        return self.start <= t < self.end
    
    def overlaps(self, other: 'TimeInterval') -> bool:
        """Check if two intervals overlap"""
        return self.start < other.end and other.start < self.end
    
    def intersection(self, other: 'TimeInterval') -> Maybe['TimeInterval']:
        """Compute intersection of two intervals"""
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if start < end:
            return Maybe.some(TimeInterval(start, end))
        return Maybe.none()


# ============================================================================
# PROBABILISTIC TYPES
# ============================================================================

@dataclass
class Probability:
    """
    Probability value ∈ [0, 1].
    Forms a bounded lattice with ∨ (max) and ∧ (min).
    """
    value: float
    
    def __post_init__(self):
        assert 0.0 <= self.value <= 1.0, f"Probability must be in [0,1], got {self.value}"
    
    def complement(self) -> 'Probability':
        """Complement: P(¬A) = 1 - P(A)"""
        return Probability(1.0 - self.value)
    
    def __and__(self, other: 'Probability') -> 'Probability':
        """Conjunction (assuming independence): P(A ∧ B) = P(A) × P(B)"""
        return Probability(self.value * other.value)
    
    def __or__(self, other: 'Probability') -> 'Probability':
        """Disjunction (assuming independence): P(A ∨ B) = P(A) + P(B) - P(A)P(B)"""
        return Probability(self.value + other.value - self.value * other.value)


@dataclass
class Distribution(Generic[T]):
    """
    Probability distribution over type T.
    Supports inference and sampling.
    """
    support: List[T]
    probabilities: NDArray[np.float64]
    
    def __post_init__(self):
        assert len(self.support) == len(self.probabilities)
        assert np.isclose(self.probabilities.sum(), 1.0), "Probabilities must sum to 1"
        assert np.all(self.probabilities >= 0), "Probabilities must be non-negative"
    
    def sample(self, rng: np.random.Generator) -> T:
        """Sample from the distribution"""
        idx = rng.choice(len(self.support), p=self.probabilities)
        return self.support[idx]
    
    def prob(self, value: T) -> Probability:
        """Get probability of specific value"""
        try:
            idx = self.support.index(value)
            return Probability(float(self.probabilities[idx]))
        except ValueError:
            return Probability(0.0)
    
    def map(self, f: Callable[[T], U]) -> 'Distribution[U]':
        """Functor map: apply function to support"""
        new_support_dict: Dict[U, float] = {}
        for val, prob in zip(self.support, self.probabilities):
            new_val = f(val)
            new_support_dict[new_val] = new_support_dict.get(new_val, 0.0) + prob
        
        new_support = list(new_support_dict.keys())
        new_probs = np.array([new_support_dict[v] for v in new_support])
        return Distribution(new_support, new_probs)


# ============================================================================
# INFORMATION-THEORETIC TYPES
# ============================================================================

@dataclass
class Entropy:
    """
    Shannon entropy H(X) = -Σ p(x) log p(x)
    Measured in bits (log base 2) or nats (log base e)
    """
    value: float
    base: str = "bits"  # "bits" or "nats"
    
    def __post_init__(self):
        assert self.value >= 0, "Entropy must be non-negative"
        assert self.base in ("bits", "nats")
    
    def to_bits(self) -> 'Entropy':
        if self.base == "bits":
            return self
        return Entropy(self.value / np.log(2), "bits")
    
    def to_nats(self) -> 'Entropy':
        if self.base == "nats":
            return self
        return Entropy(self.value * np.log(2), "nats")


@dataclass
class MutualInformation:
    """
    Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
    Measures statistical dependence between variables
    """
    value: float
    
    def __post_init__(self):
        assert self.value >= 0, "Mutual information must be non-negative"
    
    def normalized(self, h_x: Entropy, h_y: Entropy) -> float:
        """Normalized MI ∈ [0, 1]: I(X;Y) / min(H(X), H(Y))"""
        min_entropy = min(h_x.value, h_y.value)
        if min_entropy == 0:
            return 0.0
        return self.value / min_entropy


@dataclass
class KLDivergence:
    """
    Kullback-Leibler divergence D_KL(P || Q)
    Measures "distance" from Q to P (not symmetric!)
    """
    value: float
    
    def __post_init__(self):
        assert self.value >= 0, "KL divergence must be non-negative"


# ============================================================================
# GRAPH-THEORETIC TYPES
# ============================================================================

NodeId = TypeVar('NodeId')
EdgeLabel = TypeVar('EdgeLabel')


@dataclass(frozen=True)
class DirectedEdge(Generic[NodeId, EdgeLabel]):
    """Directed edge with label"""
    source: NodeId
    target: NodeId
    label: EdgeLabel
    weight: float = 1.0


@dataclass
class DirectedGraph(Generic[NodeId, EdgeLabel]):
    """
    Directed labeled graph G = (V, E, λ) where:
    - V is the set of vertices
    - E ⊆ V × V is the set of edges
    - λ: E → EdgeLabel is the labeling function
    """
    nodes: Set[NodeId]
    edges: Set[DirectedEdge[NodeId, EdgeLabel]]
    
    def successors(self, node: NodeId) -> Set[NodeId]:
        """Get all successor nodes"""
        return {e.target for e in self.edges if e.source == node}
    
    def predecessors(self, node: NodeId) -> Set[NodeId]:
        """Get all predecessor nodes"""
        return {e.source for e in self.edges if e.target == node}
    
    def in_degree(self, node: NodeId) -> int:
        """In-degree of node"""
        return len(self.predecessors(node))
    
    def out_degree(self, node: NodeId) -> int:
        """Out-degree of node"""
        return len(self.successors(node))
    
    def is_dag(self) -> bool:
        """Check if graph is a DAG (no cycles)"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: NodeId) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for successor in self.successors(node):
                if successor not in visited:
                    if has_cycle(successor):
                        return True
                elif successor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    return False
        return True
    
    def topological_sort(self) -> Maybe[List[NodeId]]:
        """Topological sort (returns None if not a DAG)"""
        if not self.is_dag():
            return Maybe.none()
        
        in_degrees = {node: self.in_degree(node) for node in self.nodes}
        queue = [node for node, deg in in_degrees.items() if deg == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for successor in self.successors(node):
                in_degrees[successor] -= 1
                if in_degrees[successor] == 0:
                    queue.append(successor)
        
        return Maybe.some(result)


# ============================================================================
# STATE SPACE TYPES
# ============================================================================

StateId = TypeVar('StateId')


@dataclass
class StateTransition(Generic[StateId]):
    """
    State transition in a state machine/Markov process.
    Represents: state_from -[action]-> state_to with probability
    """
    state_from: StateId
    state_to: StateId
    action: Optional[str] = None
    probability: Probability = field(default_factory=lambda: Probability(1.0))
    cost: float = 0.0


@dataclass
class MarkovChain(Generic[StateId]):
    """
    Discrete-time Markov chain.
    Transition matrix P where P[i,j] = P(X_{t+1} = j | X_t = i)
    """
    states: List[StateId]
    transition_matrix: NDArray[np.float64]
    
    def __post_init__(self):
        n = len(self.states)
        assert self.transition_matrix.shape == (n, n)
        # Check rows sum to 1 (stochastic matrix)
        assert np.allclose(self.transition_matrix.sum(axis=1), 1.0)
    
    def transition_prob(self, from_state: StateId, to_state: StateId) -> Probability:
        """Get transition probability"""
        i = self.states.index(from_state)
        j = self.states.index(to_state)
        return Probability(float(self.transition_matrix[i, j]))
    
    def stationary_distribution(self) -> Maybe[Distribution[StateId]]:
        """
        Compute stationary distribution π where πP = π.
        Returns None if no unique stationary distribution exists.
        """
        # Find left eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        if not np.isclose(eigenvalues[idx], 1.0):
            return Maybe.none()
        
        # Get corresponding eigenvector and normalize
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        
        if not np.all(stationary >= 0):
            return Maybe.none()
        
        return Maybe.some(Distribution(self.states, stationary))


# ============================================================================
# METRIC SPACE TYPES
# ============================================================================

@dataclass
class MetricSpace(Generic[T]):
    """
    Metric space (X, d) where d: X × X → ℝ≥0 is a distance function satisfying:
    1. d(x, y) = 0 ⟺ x = y (identity of indiscernibles)
    2. d(x, y) = d(y, x) (symmetry)
    3. d(x, z) ≤ d(x, y) + d(y, z) (triangle inequality)
    """
    elements: Set[T]
    distance: Callable[[T, T], float]
    
    def verify_metric_axioms(self, samples: List[Tuple[T, T, T]]) -> bool:
        """Verify metric axioms on sample triples"""
        for x, y, z in samples:
            d_xy = self.distance(x, y)
            d_yx = self.distance(y, x)
            d_xz = self.distance(x, z)
            d_yz = self.distance(y, z)
            
            # Symmetry
            if not np.isclose(d_xy, d_yx):
                return False
            
            # Triangle inequality
            if d_xz > d_xy + d_yz + 1e-6:
                return False
            
            # Non-negativity
            if d_xy < 0 or d_xz < 0 or d_yz < 0:
                return False
        
        return True
    
    def diameter(self) -> float:
        """Compute diameter: sup{d(x,y) : x,y ∈ X}"""
        if len(self.elements) < 2:
            return 0.0
        elements_list = list(self.elements)
        return max(
            self.distance(x, y)
            for i, x in enumerate(elements_list)
            for y in elements_list[i+1:]
        )


# ============================================================================
# TOPOLOGICAL TYPES
# ============================================================================

@dataclass
class OpenSet(Generic[T]):
    """
    Open set in a topological space.
    Represented by a characteristic function.
    """
    contains: Callable[[T], bool]
    
    def union(self, other: 'OpenSet[T]') -> 'OpenSet[T]':
        """Union of open sets is open"""
        return OpenSet(lambda x: self.contains(x) or other.contains(x))
    
    def intersection(self, other: 'OpenSet[T]') -> 'OpenSet[T]':
        """Finite intersection of open sets is open"""
        return OpenSet(lambda x: self.contains(x) and other.contains(x))


@dataclass
class TopologicalSpace(Generic[T]):
    """
    Topological space (X, τ) where τ is a topology on X.
    τ satisfies: ∅, X ∈ τ; arbitrary unions in τ; finite intersections in τ
    """
    elements: Set[T]
    topology: Set[FrozenSet[T]]  # Collection of open sets
    
    def __post_init__(self):
        # Verify topology axioms
        elements_frozen = frozenset(self.elements)
        assert frozenset() in self.topology, "Empty set must be open"
        assert elements_frozen in self.topology, "Whole space must be open"
    
    def is_open(self, subset: Set[T]) -> bool:
        """Check if subset is open"""
        return frozenset(subset) in self.topology
    
    def is_closed(self, subset: Set[T]) -> bool:
        """Check if subset is closed (complement is open)"""
        complement = self.elements - subset
        return self.is_open(complement)
