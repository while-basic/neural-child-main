# codebase_to_obsidian.py
# Description: Converts Python codebase to Obsidian documentation
# Created by: Christopher Celaya

import ast
import os
from pathlib import Path
from typing import Dict, List, Set
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodebaseParser:
    """Parses Python files to extract neural network architectures and relationships"""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
    def parse_file(self, file_path: str) -> Dict:
        """Parse a Python file and extract neural network information"""
        try:
            # Skip non-Python files and special Python files
            if not file_path.endswith('.py') or file_path.endswith('codebase_to_obsidian.py'):
                return {}
                
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
                    
            if content is None:
                print(f"Could not read {file_path} with any encoding")
                return {}
                
            tree = ast.parse(content)
            
            # Extract file information
            info = {
                'classes': [],
                'imports': [],
                'description': '',
                'relationships': []
            }
            
            # Get file description from docstring
            if (ast.get_docstring(tree)):
                info['description'] = ast.get_docstring(tree)
            
            # Process imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        info['imports'].append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        info['imports'].append(f"{node.module}.{name.name}")
                        
            # Process classes
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = self._parse_class(node)
                    info['classes'].append(class_info)
                    
                    # Extract relationships from base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            info['relationships'].append({
                                'from': node.name,
                                'to': base.id,
                                'type': 'inherits'
                            })
                        
            return info
            
        except Exception as e:
            print(f"Error parsing {file_path}: {str(e)}")
            return {}
            
    def _parse_class(self, node: ast.ClassDef) -> Dict:
        """Parse a class definition"""
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node) or '',
            'methods': [],
            'attributes': [],
            'neural_components': []
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    'name': item.name,
                    'docstring': ast.get_docstring(item) or ''
                }
                class_info['methods'].append(method_info)
                
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info['attributes'].append(target.id)
                        
            # Look for neural network components
            if isinstance(item, ast.Assign):
                value = item.value
                if isinstance(value, (ast.Call)):
                    if hasattr(value.func, 'attr'):
                        if value.func.attr in ['Linear', 'LSTM', 'GRU', 'Conv2d', 'Sequential', 'ModuleDict', 'ModuleList']:
                            class_info['neural_components'].append({
                                'type': value.func.attr,
                                'name': target.id if isinstance(target, ast.Name) else 'unnamed'
                            })
                            
        return class_info
        
    def create_obsidian_notes(self, codebase_info: Dict):
        """Create Obsidian markdown files from parsed codebase information"""
        # Create main index note
        main_content = "# Neural Child Development System\n\n"
        main_content += "This documentation is automatically generated from the codebase.\n\n"
        main_content += "## Neural Networks\n\n"
        
        for file_info in codebase_info.values():
            for class_info in file_info['classes']:
                if class_info['neural_components']:
                    main_content += f"- [[{class_info['name']}]]\n"
                    
                    # Create note for the neural network
                    self._create_network_note(class_info)
                    
        # Save main note
        main_path = self.vault_path / "index.md"
        main_path.write_text(main_content)
        
        # Create visualization note
        viz_content = "# Neural Network Visualization\n\n"
        viz_content += "```mermaid\ngraph TD\n"
        
        # Add nodes and relationships
        for file_info in codebase_info.values():
            for relationship in file_info['relationships']:
                viz_content += f"    {relationship['from']} --> {relationship['to']}\n"
                
        viz_content += "```\n"
        
        viz_path = self.vault_path / "visualization.md"
        viz_path.write_text(viz_content)
        
    def _create_network_note(self, class_info: Dict):
        """Create a note for a neural network class"""
        content = f"# {class_info['name']}\n\n"
        
        if class_info['docstring']:
            content += f"{class_info['docstring']}\n\n"
            
        content += "## Neural Components\n\n"
        for component in class_info['neural_components']:
            content += f"- {component['name']}: {component['type']}\n"
            
        content += "\n## Methods\n\n"
        for method in class_info['methods']:
            content += f"### {method['name']}\n"
            if method['docstring']:
                content += f"{method['docstring']}\n"
            content += "\n"
            
        # Save note
        note_path = self.vault_path / f"{class_info['name']}.md"
        note_path.write_text(content)

class CodebaseWatcher(FileSystemEventHandler):
    """Watches for changes in the codebase and updates documentation"""
    
    def __init__(self, codebase_path: str, vault_path: str):
        self.codebase_path = Path(codebase_path)
        self.parser = CodebaseParser(vault_path)
        
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f"Detected change in {event.src_path}")
            self.update_documentation()
            
    def update_documentation(self):
        """Update Obsidian documentation from current codebase state"""
        codebase_info = {}
        
        # Parse all Python files
        for file_path in self.codebase_path.rglob('*.py'):
            if not file_path.name.startswith('_'):  # Skip __init__.py etc.
                info = self.parser.parse_file(str(file_path))
                if info:
                    codebase_info[file_path.name] = info
                    
        # Create/update Obsidian notes
        self.parser.create_obsidian_notes(codebase_info)
        print("Updated Obsidian documentation")

def main():
    # Paths
    CODEBASE_PATH = "."  # Current directory
    VAULT_PATH = "codebase_vault"
    
    # Create watcher
    watcher = CodebaseWatcher(CODEBASE_PATH, VAULT_PATH)
    
    # Do initial documentation generation
    watcher.update_documentation()
    
    # Set up file system observer
    observer = Observer()
    observer.schedule(watcher, CODEBASE_PATH, recursive=True)
    observer.start()
    
    print(f"Watching for changes in {CODEBASE_PATH}")
    print(f"Documentation will be updated in {VAULT_PATH}")
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped watching for changes")
        
    observer.join()

if __name__ == "__main__":
    main() 