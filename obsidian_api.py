# obsidian_api.py
# Description: API wrapper for interacting with local Obsidian vault
# Created by: Christopher Celaya

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import json
import os
from datetime import datetime
import re

class ObsidianAPI:
    """API wrapper for interacting with local Obsidian vault"""
    
    def __init__(self, vault_path: Optional[str] = None):
        """Initialize Obsidian API with path to vault"""
        # Use current directory if no vault path provided
        self.vault_path = Path(vault_path) if vault_path else Path.cwd() / "obsidian_vault"
        
        # Create vault directory if it doesn't exist
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
        self.vocabulary_cache = {}
        self.last_sync = None
        
        # Create vocabulary note if it doesn't exist
        self.vocab_file = self.vault_path / "Vocabulary" / "learned_vocabulary.md"
        self.vocab_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.vocab_file.exists():
            self._initialize_vocabulary_file()
            
    def _initialize_vocabulary_file(self):
        """Create initial vocabulary file"""
        initial_content = """# Neural Child Learned Vocabulary

## Overview
This file tracks the vocabulary learned by the neural child system.

## Vocabulary List
Words are listed with their learning timestamp and emotional context.

"""
        self.vocab_file.write_text(initial_content)
            
    def write_note(self, relative_path: str, content: str):
        """Write a new note to the vault"""
        try:
            full_path = self.vault_path / relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            print(f"Wrote note to {relative_path}")
        except Exception as e:
            print(f"Error writing note {relative_path}: {str(e)}")
            
    def list_vault_files(self) -> List[str]:
        """Get list of all markdown files in the vault"""
        try:
            files = []
            for file in self.vault_path.rglob("*.md"):
                files.append(str(file.relative_to(self.vault_path)))
            print(f"Found {len(files)} markdown files in vault")
            return files
        except Exception as e:
            print(f"Error listing vault files: {str(e)}")
            return []
            
    def get_vault_file(self, file_path: str) -> str:
        """Get contents of a specific file"""
        try:
            full_path = self.vault_path / file_path
            if not full_path.exists():
                print(f"File does not exist: {file_path}")
                return ""
            return full_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error getting file {file_path}: {str(e)}")
            return ""
            
    def search_vault(self, query: str) -> List[Dict[str, Any]]:
        """Search vault for specific content"""
        try:
            results = []
            for file in self.vault_path.rglob("*.md"):
                content = file.read_text(encoding='utf-8')
                if query.lower() in content.lower():
                    results.append({
                        "file": str(file.relative_to(self.vault_path)),
                        "matches": content.lower().count(query.lower())
                    })
            return results
        except Exception as e:
            print(f"Error searching vault: {str(e)}")
            return []
            
    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data showing connections between notes"""
        try:
            nodes = []
            edges = []
            files = self.list_vault_files()
            
            # Create nodes for each file
            for file in files:
                nodes.append({
                    "id": file,
                    "label": Path(file).stem
                })
                
                # Find links in file content
                content = self.get_vault_file(file)
                links = self._extract_links(content)
                
                # Create edges for each link
                for link in links:
                    edges.append({
                        "source": file,
                        "target": f"{link}.md"
                    })
                    
            return {
                "nodes": nodes,
                "edges": edges
            }
        except Exception as e:
            print(f"Error getting graph data: {str(e)}")
            return {"nodes": [], "edges": []}
            
    def _extract_links(self, content: str) -> List[str]:
        """Extract wiki-style links from content"""
        # Match both [[Link]] and [[Link|Alias]] formats
        links = re.findall(r'\[\[(.*?)(?:\|.*?)?\]\]', content)
        return links 

    def add_vocabulary(self, word: str, emotional_context: Dict[str, float]) -> bool:
        """Add a new word to the vocabulary with emotional context"""
        try:
            if not self.vocab_file.exists():
                self._initialize_vocabulary_file()
                
            # Format emotional context
            emotion_str = ", ".join([f"{k}: {v:.2f}" for k, v in emotional_context.items()])
            
            # Create vocabulary entry
            entry = f"\n### {word}\n"
            entry += f"- Learned: {datetime.now().isoformat()}\n"
            entry += f"- Emotional Context: {emotion_str}\n"
            
            # Append to file
            with self.vocab_file.open('a') as f:
                f.write(entry)
                
            # Update cache
            self.vocabulary_cache[word] = {
                'timestamp': datetime.now().isoformat(),
                'emotional_context': emotional_context
            }
            
            return True
            
        except Exception as e:
            print(f"Error adding vocabulary: {str(e)}")
            return False
            
    def get_vocabulary(self) -> Dict[str, Dict]:
        """Get all learned vocabulary with metadata"""
        try:
            if not self.vocab_file.exists():
                return {}
                
            # Check if we need to refresh cache
            if not self.last_sync or self.vocab_file.stat().st_mtime > self.last_sync.timestamp():
                self._refresh_cache()
                
            return self.vocabulary_cache
            
        except Exception as e:
            print(f"Error getting vocabulary: {str(e)}")
            return {}
            
    def _refresh_cache(self):
        """Refresh vocabulary cache from file"""
        try:
            content = self.vocab_file.read_text()
            
            # Parse vocabulary entries
            word_pattern = r"### (\w+)\n- Learned: ([\d\-T:\.]+)\n- Emotional Context: (.+)"
            matches = re.finditer(word_pattern, content)
            
            new_cache = {}
            for match in matches:
                word = match.group(1)
                timestamp = match.group(2)
                emotion_str = match.group(3)
                
                # Parse emotional context
                emotions = {}
                for emotion_pair in emotion_str.split(", "):
                    name, value = emotion_pair.split(": ")
                    emotions[name] = float(value)
                    
                new_cache[word] = {
                    'timestamp': timestamp,
                    'emotional_context': emotions
                }
                
            self.vocabulary_cache = new_cache
            self.last_sync = datetime.now()
            
        except Exception as e:
            print(f"Error refreshing cache: {str(e)}")
            
    def get_vocabulary_size(self) -> int:
        """Get the total number of learned words"""
        return len(self.get_vocabulary()) 