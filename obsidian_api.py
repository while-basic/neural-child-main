# obsidian_api.py
# Description: Simple Obsidian API for testing the neural child system
# Created by: Christopher Celaya

import os
from pathlib import Path
from datetime import datetime
import json
import yaml
import re
from typing import Dict, List, Any, Optional

class ObsidianAPI:
    def __init__(self, vault_path: str):
        """Initialize Obsidian API with vault path.
        
        Args:
            vault_path (str): Path to Obsidian vault
        """
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create necessary directories
        self.code_dir = self.vault_path / "Code"
        self.code_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata cache
        self.metadata_cache = {}
        
    def create_note(self, filename: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new note in the Obsidian vault.
        
        Args:
            filename (str): Name of the file to create
            content (str): Content of the note
            metadata (Optional[Dict[str, Any]]): YAML frontmatter metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = self.vault_path / filename
            
            # Add YAML frontmatter if metadata is provided
            if metadata:
                metadata.update({
                    'created': datetime.now().isoformat(),
                    'last_modified': datetime.now().isoformat()
                })
                content = f"---\n{yaml.dump(metadata)}---\n\n{content}"
                
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Update metadata cache
            self.metadata_cache[filename] = metadata or {}
                
            return True
        except Exception as e:
            print(f"Error creating note: {str(e)}")
            return False
            
    def get_note(self, filename: str) -> str:
        """Get content of a note from the vault.
        
        Args:
            filename (str): Name of the file to read
            
        Returns:
            str: Content of the note
        """
        try:
            file_path = self.vault_path / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading note: {str(e)}")
            return ""
            
    def get_metadata(self, filename: str) -> Dict[str, Any]:
        """Get metadata from a note's YAML frontmatter.
        
        Args:
            filename (str): Name of the file to read
            
        Returns:
            Dict[str, Any]: Metadata from frontmatter
        """
        try:
            content = self.get_note(filename)
            if not content:
                return {}
                
            # Extract YAML frontmatter
            match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if match:
                return yaml.safe_load(match.group(1))
            return {}
            
        except Exception as e:
            print(f"Error reading metadata: {str(e)}")
            return {}
            
    def update_metadata(self, filename: str, metadata: Dict[str, Any]) -> bool:
        """Update a note's metadata.
        
        Args:
            filename (str): Name of the file to update
            metadata (Dict[str, Any]): New metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            content = self.get_note(filename)
            if not content:
                return False
                
            # Update metadata
            metadata.update({'last_modified': datetime.now().isoformat()})
            
            # Replace or add frontmatter
            new_frontmatter = f"---\n{yaml.dump(metadata)}---\n"
            if re.match(r'^---\n.*?\n---\n', content, re.DOTALL):
                content = re.sub(r'^---\n.*?\n---\n', new_frontmatter, content, flags=re.DOTALL)
            else:
                content = new_frontmatter + content
                
            # Update file
            return self.create_note(filename, content)
            
        except Exception as e:
            print(f"Error updating metadata: {str(e)}")
            return False
            
    def list_notes(self) -> list:
        """List all notes in the vault.
        
        Returns:
            list: List of filenames
        """
        try:
            return [f.name for f in self.vault_path.glob("**/*.md")]
        except Exception as e:
            print(f"Error listing notes: {str(e)}")
            return []
            
    def update_note(self, filename: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing note in the vault.
        
        Args:
            filename (str): Name of the file to update
            content (str): New content
            metadata (Optional[Dict[str, Any]]): New metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if metadata:
            return self.create_note(filename, content, metadata)
        return self.create_note(filename, content)
        
    def delete_note(self, filename: str) -> bool:
        """Delete a note from the vault.
        
        Args:
            filename (str): Name of the file to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = self.vault_path / filename
            if file_path.exists():
                file_path.unlink()
                # Remove from metadata cache
                self.metadata_cache.pop(filename, None)
                return True
            return False
        except Exception as e:
            print(f"Error deleting note: {str(e)}")
            return False
            
    def add_tag(self, filename: str, tag: str) -> bool:
        """Add a tag to an existing note.
        
        Args:
            filename (str): Name of the file to tag
            tag (str): Tag to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metadata = self.get_metadata(filename)
            if not metadata:
                metadata = {}
                
            # Add tag if not already present
            tags = metadata.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
            if tag not in tags:
                tags.append(tag)
                metadata['tags'] = tags
                return self.update_metadata(filename, metadata)
            return True
        except Exception as e:
            print(f"Error adding tag: {str(e)}")
            return False
            
    def add_connection(self, source: str, target: str, connection_type: str = "related") -> bool:
        """Add a connection between two notes.
        
        Args:
            source (str): Source note filename
            target (str): Target note filename
            connection_type (str): Type of connection
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update source metadata
            source_metadata = self.get_metadata(source)
            connections = source_metadata.get('connections', [])
            if isinstance(connections, str):
                connections = [c.strip() for c in connections.split(',')]
            connection = {'target': target, 'type': connection_type}
            if connection not in connections:
                connections.append(connection)
                source_metadata['connections'] = connections
                self.update_metadata(source, source_metadata)
                
            # Update target metadata
            target_metadata = self.get_metadata(target)
            back_connections = target_metadata.get('connections', [])
            if isinstance(back_connections, str):
                back_connections = [c.strip() for c in back_connections.split(',')]
            back_connection = {'target': source, 'type': f"back_{connection_type}"}
            if back_connection not in back_connections:
                back_connections.append(back_connection)
                target_metadata['connections'] = back_connections
                self.update_metadata(target, target_metadata)
                
            return True
        except Exception as e:
            print(f"Error adding connection: {str(e)}")
            return False

    def run_command(self, command: str) -> str:
        """Run a shell command and return its output.
        
        Args:
            command (str): Command to run
            
        Returns:
            str: Command output
        """
        try:
            import subprocess
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            print(f"Error running command: {str(e)}")
            return "" 