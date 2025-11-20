"""
Interactive Documentation with Executable Examples

Literate programming documentation with executable code blocks.
Keeps documentation synchronized with code through executable examples.
"""

from typing import List, Dict
from dataclasses import dataclass

@dataclass
class CodeBlock:
    """Executable code block in documentation"""
    language: str
    code: str
    expected_output: str = None

class LiterateDocument:
    """
    Documentation with executable code blocks.
    
    Inspired by Jupyter notebooks and org-mode.
    """
    def __init__(self, markdown_path: str):
        self.markdown_path = markdown_path
        self.code_blocks = []
        self._parse_markdown()
        
    def _parse_markdown(self):
        """Extract code blocks from markdown"""
        try:
            with open(self.markdown_path) as f:
                content = f.read()
            
            in_code_block = False
            current_block = []
            
            for line in content.split('\n'):
                if line.startswith('```python'):
                    in_code_block = True
                    current_block = []
                elif line.startswith('```') and in_code_block:
                    in_code_block = False
                    self.code_blocks.append(CodeBlock(
                        language='python',
                        code='\n'.join(current_block)
                    ))
                elif in_code_block:
                    current_block.append(line)
        except FileNotFoundError:
            pass
    
    def execute_all(self) -> List[Dict]:
        """Execute all code blocks and return results"""
        results = []
        global_namespace = {}
        
        for i, block in enumerate(self.code_blocks):
            try:
                exec(block.code, global_namespace)
                results.append({'block_index': i, 'success': True, 'error': None})
            except Exception as e:
                results.append({'block_index': i, 'success': False, 'error': str(e)})
        
        return results

class DocTestRunner:
    """Run documentation examples as tests"""
    def __init__(self):
        self.docs = []
        
    def add_document(self, path: str):
        """Add document to test suite"""
        self.docs.append(LiterateDocument(path))
    
    def run_all(self) -> bool:
        """Run all documentation tests"""
        all_passed = True
        for doc in self.docs:
            results = doc.execute_all()
            for result in results:
                if not result['success']:
                    all_passed = False
        return all_passed
