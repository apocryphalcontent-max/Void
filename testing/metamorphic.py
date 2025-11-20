"""
Metamorphic Testing for Tools

Tests tools using metamorphic relations - properties that should hold
across related inputs.
"""

from typing import Callable, Any, List, Tuple

class MetamorphicRelation:
    """A metamorphic relation between test inputs/outputs"""
    def __init__(self, name: str, transform: Callable, check: Callable):
        self.name = name
        self.transform = transform  # Transform input
        self.check = check  # Check relation holds
        
class MetamorphicTester:
    """Executes metamorphic tests"""
    def __init__(self, tool: Any):
        self.tool = tool
        self.relations: List[MetamorphicRelation] = []
        
    def add_relation(self, relation: MetamorphicRelation):
        """Add metamorphic relation"""
        self.relations.append(relation)
    
    def test(self, input_data: Any) -> bool:
        """Test all metamorphic relations"""
        original_output = self.tool.execute(input_data)
        
        for relation in self.relations:
            # Transform input
            transformed_input = relation.transform(input_data)
            transformed_output = self.tool.execute(transformed_input)
            
            # Check relation holds
            if not relation.check(original_output, transformed_output):
                return False
        
        return True

# Example metamorphic relations
def permutation_invariance(original, permuted):
    """Output should be same for permuted input"""
    return sorted(original) == sorted(permuted)

def additive_relation(output1, output2, combined_output):
    """f(x) + f(y) should equal f(x+y) for additive functions"""
    return abs((output1 + output2) - combined_output) < 1e-6
