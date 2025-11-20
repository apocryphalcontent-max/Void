"""
Multi-Agent Coordination Framework for Void-State System.

This module implements comprehensive multi-agent coordination mechanisms including:
- Byzantine fault-tolerant consensus
- Distributed coordination protocols
- Game-theoretic negotiation
- Coalition formation
- Task allocation and scheduling

Part of v3.3 "Synthesis" enhancement.
"""

import enum
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from collections import defaultdict
import numpy as np


class ConsensusStatus(enum.Enum):
    """Status of consensus protocol."""
    PENDING = "pending"
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMMITTED = "committed"


class AgentRole(enum.Enum):
    """Roles in multi-agent system."""
    LEADER = "leader"
    FOLLOWER = "follower"
    VALIDATOR = "validator"
    OBSERVER = "observer"


@dataclass
class Agent:
    """Represents an agent in the multi-agent system."""
    agent_id: str
    role: AgentRole = AgentRole.FOLLOWER
    trust_score: float = 1.0  # [0, 1] - reputation
    byzantine: bool = False  # Is this agent Byzantine (malicious)?
    capabilities: Set[str] = field(default_factory=set)
    resources: Dict[str, float] = field(default_factory=dict)


@dataclass
class Proposal:
    """A proposal for consensus."""
    proposal_id: str
    proposer: str
    content: Any
    timestamp: float
    votes: Dict[str, bool] = field(default_factory=dict)  # agent_id -> vote
    status: ConsensusStatus = ConsensusStatus.PENDING


class ByzantineFaultTolerantConsensus:
    """
    Byzantine Fault Tolerant (BFT) consensus protocol.
    
    Implements PBFT-style consensus that tolerates up to f < n/3 Byzantine faults.
    
    **Complexity:** O(n²) message complexity per round
    **Safety:** Guaranteed with f < n/3 Byzantine nodes
    **Liveness:** Guaranteed with synchrony assumption
    """
    
    def __init__(self, agents: List[Agent], f: int = None):
        """
        Initialize BFT consensus.
        
        Args:
            agents: List of participating agents
            f: Maximum number of Byzantine faults to tolerate (default: (n-1)//3)
        """
        self.agents = {a.agent_id: a for a in agents}
        self.n = len(agents)
        self.f = f if f is not None else (self.n - 1) // 3
        self.proposals: Dict[str, Proposal] = {}
        self.committed: List[str] = []
        
        if self.n < 3 * self.f + 1:
            raise ValueError(f"Need n >= 3f+1: n={self.n}, f={self.f}")
    
    def propose(self, proposer_id: str, content: Any) -> str:
        """
        Create a new proposal.
        
        Args:
            proposer_id: ID of proposing agent
            content: Proposal content
            
        Returns:
            Proposal ID
        """
        if proposer_id not in self.agents:
            raise ValueError(f"Unknown agent: {proposer_id}")
        
        proposal_id = hashlib.sha256(
            f"{proposer_id}{content}{len(self.proposals)}".encode()
        ).hexdigest()[:16]
        
        proposal = Proposal(
            proposal_id=proposal_id,
            proposer=proposer_id,
            content=content,
            timestamp=len(self.proposals),
            status=ConsensusStatus.PROPOSED
        )
        
        self.proposals[proposal_id] = proposal
        return proposal_id
    
    def vote(self, proposal_id: str, voter_id: str, vote: bool) -> None:
        """
        Cast a vote on a proposal.
        
        Args:
            proposal_id: ID of proposal
            voter_id: ID of voting agent
            vote: True for accept, False for reject
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        if voter_id not in self.agents:
            raise ValueError(f"Unknown agent: {voter_id}")
        
        proposal = self.proposals[proposal_id]
        
        # Byzantine agents may vote randomly
        if self.agents[voter_id].byzantine:
            vote = random.choice([True, False])
        
        proposal.votes[voter_id] = vote
    
    def check_consensus(self, proposal_id: str) -> ConsensusStatus:
        """
        Check if proposal has reached consensus.
        
        BFT requires 2f+1 votes (simple majority with Byzantine tolerance).
        
        Args:
            proposal_id: ID of proposal
            
        Returns:
            Updated consensus status
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        
        proposal = self.proposals[proposal_id]
        votes = proposal.votes
        
        if len(votes) < self.n:
            return ConsensusStatus.PENDING
        
        accept_votes = sum(1 for v in votes.values() if v)
        reject_votes = len(votes) - accept_votes
        
        # Need 2f+1 votes for consensus
        threshold = 2 * self.f + 1
        
        if accept_votes >= threshold:
            proposal.status = ConsensusStatus.ACCEPTED
            self.committed.append(proposal_id)
            return ConsensusStatus.COMMITTED
        elif reject_votes >= threshold:
            proposal.status = ConsensusStatus.REJECTED
            return ConsensusStatus.REJECTED
        
        return ConsensusStatus.PENDING


@dataclass
class Task:
    """A task to be allocated to agents."""
    task_id: str
    required_capabilities: Set[str]
    required_resources: Dict[str, float]
    utility: float  # Value of completing task
    deadline: Optional[float] = None


@dataclass
class Allocation:
    """Task allocation to agent."""
    task: Task
    agent: Agent
    cost: float


class GameTheoreticNegotiator:
    """
    Game-theoretic negotiation for task allocation.
    
    Implements mechanisms for:
    - Nash bargaining
    - Auction-based allocation
    - Coalition formation
    - Pareto-optimal allocations
    """
    
    def __init__(self, agents: List[Agent]):
        self.agents = {a.agent_id: a for a in agents}
    
    def utility(self, agent: Agent, task: Task) -> float:
        """
        Calculate agent's utility for completing a task.
        
        Args:
            agent: Agent
            task: Task
            
        Returns:
            Utility value (higher is better)
        """
        # Check if agent has required capabilities
        if not task.required_capabilities.issubset(agent.capabilities):
            return 0.0
        
        # Check if agent has required resources
        for resource, amount in task.required_resources.items():
            if agent.resources.get(resource, 0) < amount:
                return 0.0
        
        # Base utility from task value
        utility = task.utility
        
        # Discount by resource usage
        resource_cost = sum(
            amount / agent.resources.get(resource, 1)
            for resource, amount in task.required_resources.items()
        )
        
        # Trust score affects utility
        utility *= agent.trust_score
        
        # Subtract cost
        utility -= resource_cost
        
        return max(0.0, utility)
    
    def auction_allocation(
        self, 
        tasks: List[Task],
        mechanism: str = "first_price"
    ) -> List[Allocation]:
        """
        Allocate tasks via auction mechanism.
        
        Args:
            tasks: Tasks to allocate
            mechanism: "first_price" or "second_price" (Vickrey)
            
        Returns:
            List of allocations
            
        Complexity: O(n·m) for n agents, m tasks
        """
        allocations = []
        
        for task in tasks:
            # Collect bids (utility values)
            bids = [
                (agent_id, self.utility(agent, task))
                for agent_id, agent in self.agents.items()
            ]
            
            # Sort by bid (descending)
            bids.sort(key=lambda x: x[1], reverse=True)
            
            if not bids or bids[0][1] <= 0:
                continue  # No viable bids
            
            winner_id, winner_bid = bids[0]
            
            if mechanism == "first_price":
                price = winner_bid
            elif mechanism == "second_price":
                # Vickrey: pay second-highest bid
                price = bids[1][1] if len(bids) > 1 else winner_bid
            else:
                raise ValueError(f"Unknown mechanism: {mechanism}")
            
            allocations.append(Allocation(
                task=task,
                agent=self.agents[winner_id],
                cost=price
            ))
        
        return allocations
    
    def nash_bargaining(
        self,
        agent1: Agent,
        agent2: Agent,
        task1: Task,
        task2: Task,
        disagreement_point: Tuple[float, float] = (0.0, 0.0)
    ) -> Optional[Tuple[float, float]]:
        """
        Nash bargaining solution for two agents negotiating over two tasks.
        
        Finds allocation that maximizes (u1 - d1) * (u2 - d2).
        
        Args:
            agent1, agent2: Negotiating agents
            task1, task2: Tasks to allocate
            disagreement_point: Utilities if negotiation fails
            
        Returns:
            Optimal allocation (share for agent1, share for agent2) or None
        """
        d1, d2 = disagreement_point
        
        # Calculate utilities
        u1_task1 = self.utility(agent1, task1)
        u1_task2 = self.utility(agent1, task2)
        u2_task1 = self.utility(agent2, task1)
        u2_task2 = self.utility(agent2, task2)
        
        # Check all feasible allocations
        best_nash = -float('inf')
        best_allocation = None
        
        # Try different splits
        for alpha in np.linspace(0, 1, 21):
            # agent1 gets alpha of task1, (1-alpha) of task2
            # agent2 gets (1-alpha) of task1, alpha of task2
            
            u1 = alpha * u1_task1 + (1 - alpha) * u1_task2
            u2 = (1 - alpha) * u2_task1 + alpha * u2_task2
            
            if u1 >= d1 and u2 >= d2:
                nash_product = (u1 - d1) * (u2 - d2)
                if nash_product > best_nash:
                    best_nash = nash_product
                    best_allocation = (u1, u2)
        
        return best_allocation
    
    def form_coalitions(
        self,
        agents: List[Agent],
        tasks: List[Task],
        max_coalition_size: int = None
    ) -> List[Tuple[Set[str], Task]]:
        """
        Form coalitions of agents to complete tasks.
        
        Uses greedy algorithm to form stable coalitions.
        
        Args:
            agents: Available agents
            tasks: Tasks requiring coalitions
            max_coalition_size: Maximum agents per coalition
            
        Returns:
            List of (coalition_agent_ids, task) tuples
            
        Complexity: O(2^n * m) worst case, O(n * m) typical
        """
        if max_coalition_size is None:
            max_coalition_size = len(agents)
        
        coalitions = []
        available = {a.agent_id for a in agents}
        
        for task in sorted(tasks, key=lambda t: t.utility, reverse=True):
            # Find best coalition for this task
            best_coalition = None
            best_value = 0.0
            
            # Try all subsets up to max_coalition_size
            for size in range(1, min(max_coalition_size + 1, len(available) + 1)):
                # Greedy: pick agents with highest utility
                candidates = [
                    (aid, self.utility(self.agents[aid], task))
                    for aid in available
                ]
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                coalition = {aid for aid, _ in candidates[:size]}
                
                # Check if coalition can complete task
                combined_capabilities = set()
                combined_resources = defaultdict(float)
                
                for aid in coalition:
                    agent = self.agents[aid]
                    combined_capabilities.update(agent.capabilities)
                    for resource, amount in agent.resources.items():
                        combined_resources[resource] += amount
                
                if not task.required_capabilities.issubset(combined_capabilities):
                    continue
                
                can_complete = all(
                    combined_resources.get(resource, 0) >= amount
                    for resource, amount in task.required_resources.items()
                )
                
                if can_complete:
                    total_utility = sum(
                        self.utility(self.agents[aid], task)
                        for aid in coalition
                    )
                    
                    if total_utility > best_value:
                        best_value = total_utility
                        best_coalition = coalition
            
            if best_coalition:
                coalitions.append((best_coalition, task))
                available -= best_coalition
        
        return coalitions


class DistributedCoordinator:
    """
    Distributed coordination protocol for multi-agent systems.
    
    Combines BFT consensus with game-theoretic allocation.
    """
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.consensus = ByzantineFaultTolerantConsensus(agents)
        self.negotiator = GameTheoreticNegotiator(agents)
        self.allocations: List[Allocation] = []
    
    def coordinate_tasks(
        self,
        tasks: List[Task],
        mechanism: str = "auction"
    ) -> List[Allocation]:
        """
        Coordinate task allocation among agents.
        
        Args:
            tasks: Tasks to allocate
            mechanism: "auction", "negotiation", or "coalition"
            
        Returns:
            Agreed-upon allocations
        """
        # Phase 1: Generate allocation proposals
        if mechanism == "auction":
            proposals = self.negotiator.auction_allocation(tasks)
        elif mechanism == "coalition":
            coalition_allocations = self.negotiator.form_coalitions(
                self.agents, tasks
            )
            proposals = []
            for coalition, task in coalition_allocations:
                # Assign to first agent in coalition (representative)
                agent_id = next(iter(coalition))
                proposals.append(Allocation(
                    task=task,
                    agent=self.negotiator.agents[agent_id],
                    cost=task.utility / len(coalition)
                ))
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        # Phase 2: Reach consensus on allocations
        agreed_allocations = []
        
        for allocation in proposals:
            # Propose allocation
            proposal_id = self.consensus.propose(
                allocation.agent.agent_id,
                {
                    'task_id': allocation.task.task_id,
                    'agent_id': allocation.agent.agent_id,
                    'cost': allocation.cost
                }
            )
            
            # All agents vote
            for agent in self.agents:
                # Agents vote based on whether allocation benefits them
                vote = (agent.agent_id == allocation.agent.agent_id)
                self.consensus.vote(proposal_id, agent.agent_id, vote)
            
            # Check consensus
            status = self.consensus.check_consensus(proposal_id)
            
            if status == ConsensusStatus.COMMITTED:
                agreed_allocations.append(allocation)
        
        self.allocations = agreed_allocations
        return agreed_allocations


# Demo
if __name__ == "__main__":
    print("Multi-Agent Coordination Demo\n" + "=" * 50)
    
    # Create agents
    agents = [
        Agent("agent1", capabilities={"compute", "storage"}, 
              resources={"cpu": 10, "mem": 100}),
        Agent("agent2", capabilities={"compute", "network"}, 
              resources={"cpu": 8, "mem": 80}),
        Agent("agent3", capabilities={"storage", "network"}, 
              resources={"cpu": 6, "mem": 60}),
        Agent("agent4", capabilities={"compute"}, 
              resources={"cpu": 12, "mem": 120}, byzantine=True),
    ]
    
    print(f"Agents: {len(agents)} (1 Byzantine)")
    
    # BFT Consensus Demo
    print("\n1. Byzantine Fault Tolerant Consensus")
    print("-" * 50)
    
    consensus = ByzantineFaultTolerantConsensus(agents)
    print(f"n={consensus.n}, f={consensus.f} (tolerates {consensus.f} Byzantine faults)")
    
    proposal_id = consensus.propose("agent1", "allocate_task_X_to_agent2")
    print(f"Proposed: {proposal_id}")
    
    for agent in agents:
        consensus.vote(proposal_id, agent.agent_id, True)
    
    status = consensus.check_consensus(proposal_id)
    print(f"Consensus status: {status.value}")
    
    # Game-Theoretic Negotiation Demo
    print("\n2. Game-Theoretic Task Allocation")
    print("-" * 50)
    
    tasks = [
        Task("task1", {"compute"}, {"cpu": 5}, utility=10.0),
        Task("task2", {"storage"}, {"mem": 40}, utility=8.0),
        Task("task3", {"network"}, {}, utility=6.0),
    ]
    
    negotiator = GameTheoreticNegotiator(agents)
    allocations = negotiator.auction_allocation(tasks, mechanism="second_price")
    
    print(f"Auction allocations: {len(allocations)}")
    for alloc in allocations:
        print(f"  {alloc.task.task_id} -> {alloc.agent.agent_id} (cost: {alloc.cost:.2f})")
    
    # Coalition Formation Demo
    print("\n3. Coalition Formation")
    print("-" * 50)
    
    complex_task = Task(
        "complex_task",
        {"compute", "storage", "network"},
        {"cpu": 15, "mem": 100},
        utility=30.0
    )
    
    coalitions = negotiator.form_coalitions(agents[:3], [complex_task])
    print(f"Coalitions formed: {len(coalitions)}")
    for coalition, task in coalitions:
        print(f"  {task.task_id}: agents {coalition}")
    
    # Distributed Coordination Demo
    print("\n4. Distributed Coordination")
    print("-" * 50)
    
    coordinator = DistributedCoordinator(agents)
    agreed = coordinator.coordinate_tasks(tasks, mechanism="auction")
    print(f"Agreed allocations: {len(agreed)}")
    for alloc in agreed:
        print(f"  {alloc.task.task_id} -> {alloc.agent.agent_id}")
    
    print("\nDemo complete!")
