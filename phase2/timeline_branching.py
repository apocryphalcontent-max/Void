"""
Timeline Branching Engine (Phase 2, Layer 3)

Enables creation and exploration of alternative execution timelines.
Supports speculation, rollback, and comparison of different execution paths.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

@dataclass
class TimelineState:
    """State at a point in timeline"""
    timestamp: float
    tool_states: Dict[str, Any]
    metrics: Dict[str, float]

class Timeline:
    """Represents an execution timeline"""
    def __init__(self, timeline_id: str, parent_id: Optional[str] = None):
        self.timeline_id = timeline_id
        self.parent_id = parent_id
        self.states: List[TimelineState] = []
        self.branch_point = None
        
    def record_state(self, state: TimelineState):
        """Record state in timeline"""
        self.states.append(state)
    
    def get_state_at(self, timestamp: float) -> Optional[TimelineState]:
        """Get state at specific timestamp"""
        for state in reversed(self.states):
            if state.timestamp <= timestamp:
                return state
        return None

class TimelineBranchingEngine:
    """
    Engine for managing timeline branches.
    
    Supports speculative execution and exploration of alternative paths.
    """
    def __init__(self):
        self.timelines: Dict[str, Timeline] = {}
        self.main_timeline = Timeline("main")
        self.timelines["main"] = self.main_timeline
        self.active_timeline = self.main_timeline
        
    def create_branch(self, branch_id: str, from_timeline: str = "main",
                     from_timestamp: float = None) -> Timeline:
        """
        Create new timeline branch from existing timeline.
        
        Args:
            branch_id: ID for new branch
            from_timeline: Timeline to branch from
            from_timestamp: Point in time to branch from
            
        Returns:
            New timeline branch
        """
        parent = self.timelines[from_timeline]
        branch = Timeline(branch_id, parent_id=from_timeline)
        
        # Copy state from parent at branch point
        if from_timestamp:
            state = parent.get_state_at(from_timestamp)
            if state:
                branch.record_state(state)
                branch.branch_point = from_timestamp
        
        self.timelines[branch_id] = branch
        return branch
    
    def switch_timeline(self, timeline_id: str):
        """Switch to different timeline"""
        if timeline_id in self.timelines:
            self.active_timeline = self.timelines[timeline_id]
    
    def merge_timelines(self, source_id: str, target_id: str):
        """Merge source timeline into target timeline"""
        source = self.timelines.get(source_id)
        target = self.timelines.get(target_id)
        
        if source and target:
            # Merge states from source into target
            for state in source.states:
                if state.timestamp > (target.states[-1].timestamp if target.states else 0):
                    target.record_state(state)
    
    def compare_timelines(self, timeline1_id: str, timeline2_id: str) -> Dict[str, Any]:
        """Compare two timelines"""
        t1 = self.timelines.get(timeline1_id)
        t2 = self.timelines.get(timeline2_id)
        
        if not t1 or not t2:
            return {}
        
        return {
            'state_count_diff': len(t1.states) - len(t2.states),
            'timeline1_states': len(t1.states),
            'timeline2_states': len(t2.states)
        }

# Applications:
# - Speculative execution of tools
# - A/B testing of tool configurations
# - Rollback to previous states
# - Explore "what-if" scenarios
