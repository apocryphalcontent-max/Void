"""
Natural Language Tool Specification

Allows tools to be specified in natural language, compiled to formal specs.
Lowers barrier to entry and enables rapid prototyping.
"""

from typing import Dict, List
import re

class NLSpecificationParser:
    """
    Parse natural language into formal tool specification.
    
    Uses pattern matching and NLP to extract structure from text descriptions.
    """
    def __init__(self):
        self.templates = {}
        
    def parse(self, nl_text: str) -> Dict:
        """
        Parse natural language specification.
        
        Example input:
        "Create a tool that detects anomalies in memory usage.
         Inputs: memory_samples (list of floats), threshold (float, default: 3.0)
         Outputs: anomalies (list of indices)
         Behavior: Compute z-score for each sample and flag if |z-score| > threshold"
        """
        sections = self._extract_sections(nl_text)
        
        spec = {
            'name': self._infer_name(sections.get('description', '')),
            'inputs': self._parse_io_section(sections.get('inputs', '')),
            'outputs': self._parse_io_section(sections.get('outputs', '')),
            'behavior': sections.get('behavior', ''),
            'constraints': self._parse_constraints(sections.get('constraints', ''))
        }
        
        return spec
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract structured sections from NL text"""
        sections = {}
        current_section = 'description'
        current_content = []
        
        for line in text.split('\n'):
            line = line.strip()
            
            if line.lower().startswith('inputs:'):
                sections[current_section] = '\n'.join(current_content)
                current_section = 'inputs'
                current_content = []
            elif line.lower().startswith('outputs:'):
                sections[current_section] = '\n'.join(current_content)
                current_section = 'outputs'
                current_content = []
            elif line.lower().startswith('behavior:'):
                sections[current_section] = '\n'.join(current_content)
                current_section = 'behavior'
                current_content = []
            elif line.lower().startswith('constraints:'):
                sections[current_section] = '\n'.join(current_content)
                current_section = 'constraints'
                current_content = []
            else:
                current_content.append(line)
        
        sections[current_section] = '\n'.join(current_content)
        return sections
    
    def _parse_io_section(self, text: str) -> Dict[str, str]:
        """Parse input/output section"""
        items = {}
        pattern = r'-\s*(\w+)[:\s]+(.+)'
        matches = re.findall(pattern, text)
        
        for name, description in matches:
            items[name] = description.strip()
        
        return items
    
    def _parse_constraints(self, text: str) -> List[str]:
        """Parse constraints"""
        constraints = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                constraints.append(line[1:].strip())
        return constraints
    
    def _infer_name(self, description: str) -> str:
        """Infer tool name from description"""
        words = description.lower().split()
        action_verbs = ['detect', 'analyze', 'compute', 'calculate', 'monitor']
        
        for verb in action_verbs:
            if verb in words:
                idx = words.index(verb)
                if idx + 1 < len(words):
                    return f"{verb}_{words[idx+1]}"
        
        return "custom_tool"

class SpecificationGenerator:
    """Generate implementation from specification"""
    def __init__(self):
        pass
        
    def generate_tool_code(self, spec: Dict) -> str:
        """Generate Python code implementing tool"""
        class_name = ''.join(word.capitalize() for word in spec['name'].split('_'))
        
        code = f'''
class {class_name}:
    """
    {spec.get('behavior', 'Auto-generated tool')}
    """
    def __init__(self):
        pass
    
    def execute(self, **inputs):
        """Execute tool"""
        # Generated implementation
        pass
'''
        return code

# Applications:
# - Rapid tool prototyping
# - Lower barrier to entry
# - Documentation-driven development
# - Natural specification
