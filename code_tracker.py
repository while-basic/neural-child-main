# code_tracker.py
# Description: Code tracking system for Obsidian integration
# Created by: Christopher Celaya

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import networkx as nx
from obsidian_api import ObsidianAPI

class CodeTracker:
    def __init__(self, vault_path: str = "/main_vault"):
        """Initialize the code tracker.
        
        Args:
            vault_path (str): Path to Obsidian vault
        """
        self.vault_path = Path(vault_path)
        self.api = ObsidianAPI(vault_path)
        self.code_dir = self.vault_path / "Code"
        self.code_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file and extract its structure.
        
        Args:
            file_path (str): Path to the Python file
            
        Returns:
            Dict[str, Any]: File structure including classes, methods, etc.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            structure = {
                'classes': [],
                'functions': [],
                'imports': [],
                'global_vars': []
            }
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [method.name for method in node.body if isinstance(method, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node) or "No documentation"
                    }
                    structure['classes'].append(class_info)
                    
                elif isinstance(node, ast.FunctionDef):
                    function_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node) or "No documentation"
                    }
                    structure['functions'].append(function_info)
                    
                elif isinstance(node, ast.Import):
                    structure['imports'].extend(alias.name for alias in node.names)
                    
                elif isinstance(node, ast.ImportFrom):
                    module_prefix = f"{node.module}." if node.module else ""
                    structure['imports'].extend(f"{module_prefix}{alias.name}" for alias in node.names)
                    
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            structure['global_vars'].append(target.id)
                            
            return structure
            
        except Exception as e:
            print(f"Error analyzing file {file_path}: {str(e)}")
            return {}
            
    def create_code_note(self, file_path: str, structure: Dict[str, Any]) -> bool:
        """Create an Obsidian note for a code file.
        
        Args:
            file_path (str): Path to the Python file
            structure (Dict[str, Any]): File structure
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filename = Path(file_path).stem
            note_content = [
                f"# {filename}.py",
                "",
                "## File Structure",
                "",
                "### Classes",
                ""
            ]
            
            # Add classes
            for class_info in structure['classes']:
                note_content.extend([
                    f"#### {class_info['name']}",
                    "",
                    f"```python",
                    f"class {class_info['name']}:",
                    f"    \"\"\"{class_info['docstring']}\"\"\"",
                    f"```",
                    "",
                    "Methods:",
                    "".join([f"- `{method}`\n" for method in class_info['methods']]),
                    ""
                ])
                
            # Add functions
            note_content.extend([
                "### Functions",
                ""
            ])
            
            for func_info in structure['functions']:
                note_content.extend([
                    f"#### {func_info['name']}",
                    "",
                    f"```python",
                    f"def {func_info['name']}({', '.join(func_info['args'])}):",
                    f"    \"\"\"{func_info['docstring']}\"\"\"",
                    f"```",
                    ""
                ])
                
            # Add imports and global variables
            note_content.extend([
                "### Imports",
                "",
                "".join([f"- `{imp}`\n" for imp in structure['imports']]),
                "",
                "### Global Variables",
                "",
                "".join([f"- `{var}`\n" for var in structure['global_vars']]),
                "",
                "## Connections",
                "",
                "```dataview",
                "LIST",
                f"FROM [[{filename}]] AND #code",
                "```",
                "",
                "---",
                "tags: #code #python",
                f"last_updated: {datetime.now().isoformat()}",
                "---"
            ])
            
            return self.api.create_note(
                f"Code/{filename}.md",
                "\n".join(note_content)
            )
            
        except Exception as e:
            print(f"Error creating note for {file_path}: {str(e)}")
            return False
            
    def update_code_graph(self):
        """Update the code dependency graph in Obsidian."""
        try:
            # Create graph
            graph = nx.DiGraph()
            
            # Add nodes and edges from code files
            code_files = list(self.code_dir.glob("*.md"))
            for file in code_files:
                content = self.api.get_note(f"Code/{file.name}")
                if not content:
                    continue
                    
                # Add node
                graph.add_node(file.stem)
                
                # Add edges based on imports
                imports = re.findall(r'- `(.*?)`', content)
                for imp in imports:
                    if imp in [f.stem for f in code_files]:
                        graph.add_edge(file.stem, imp)
                        
            # Create Mermaid diagram
            mermaid = ["```mermaid", "graph TD"]
            for edge in graph.edges():
                mermaid.append(f"    {edge[0]} --> {edge[1]}")
            mermaid.append("```")
            
            # Update graph note
            graph_content = [
                "# Code Dependency Graph",
                "",
                "This graph shows the dependencies between Python files in the project.",
                "",
                "\n".join(mermaid),
                "",
                "## File List",
                "",
                "".join([f"- [[Code/{node}|{node}]]\n" for node in graph.nodes()]),
                "",
                "---",
                "tags: #code #graph",
                f"last_updated: {datetime.now().isoformat()}",
                "---"
            ]
            
            return self.api.create_note(
                "Code/dependency_graph.md",
                "\n".join(graph_content)
            )
            
        except Exception as e:
            print(f"Error updating code graph: {str(e)}")
            return False
            
    def track_changes(self, file_path: str):
        """Track changes in a Python file and update Obsidian documentation.
        
        Args:
            file_path (str): Path to the Python file
        """
        try:
            # Analyze file
            structure = self.analyze_file(file_path)
            if not structure:
                return
                
            # Create/update note
            success = self.create_code_note(file_path, structure)
            if not success:
                return
                
            # Update dependency graph
            self.update_code_graph()
            
        except Exception as e:
            print(f"Error tracking changes in {file_path}: {str(e)}") 