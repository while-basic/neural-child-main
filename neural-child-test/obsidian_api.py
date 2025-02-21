# obsidian_api.py
# Description: Simple Obsidian API for testing the neural child system
# Created by: Christopher Celaya

import os
from pathlib import Path
from datetime import datetime
import json

class ObsidianAPI:
    def __init__(self, vault_path: str):
        """Initialize Obsidian API with vault path.
        
        Args:
            vault_path (str): Path to Obsidian vault
        """
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
    def create_note(self, filename: str, content: str) -> bool:
        """Create a new note in the Obsidian vault.
        
        Args:
            filename (str): Name of the file to create
            content (str): Content of the note
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = self.vault_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
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
            
    def list_notes(self) -> list:
        """List all notes in the vault.
        
        Returns:
            list: List of filenames
        """
        try:
            return [f.name for f in self.vault_path.glob("*.md")]
        except Exception as e:
            print(f"Error listing notes: {str(e)}")
            return []
            
    def update_note(self, filename: str, content: str) -> bool:
        """Update an existing note in the vault.
        
        Args:
            filename (str): Name of the file to update
            content (str): New content
            
        Returns:
            bool: True if successful, False otherwise
        """
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
            content = self.get_note(filename)
            if not content:
                return False
                
            # Add tag if not already present
            if f"#{tag}" not in content:
                content += f"\n#{tag}"
                return self.update_note(filename, content)
            return True
        except Exception as e:
            print(f"Error adding tag: {str(e)}")
            return False 