# obsidian_connector.py
# Description: Obsidian integration for neural child development
# Created by: Christopher Celaya

import os
from pathlib import Path
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
from logger import DevelopmentLogger

class ObsidianConnector:
    def __init__(self, logger: DevelopmentLogger, api_key: str, vault_name: str = "neural_child", vault_path: Optional[str] = None):
        """Initialize the Obsidian connector."""
        self.logger = logger
        self.api_key = api_key
        self.vault_name = vault_name
        
        # Set vault path
        self.vault_path = Path(vault_path) if vault_path else Path.cwd() / "obsidian_vault" / vault_name
        
        # Create vault directory structure
        self.initialize_vault()
    
    def initialize_vault(self):
        """Initialize the vault structure."""
        try:
            # Create main vault directory
            self.vault_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            folders = [
                "Memories",
                "Emotional_States",
                "Development",
                "Language_Learning",
                "Interactions"
            ]
            
            for folder in folders:
                folder_path = self.vault_path / folder
                folder_path.mkdir(parents=True, exist_ok=True)
                
            # Create .obsidian directory to mark this as an Obsidian vault
            obsidian_dir = self.vault_path / ".obsidian"
            obsidian_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a basic Obsidian config
            config = {
                "basePath": str(self.vault_path),
                "useMarkdownLinks": True,
                "newLinkFormat": "relative",
                "attachmentFolderPath": "attachments"
            }
            
            config_path = obsidian_dir / "config"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            # Create a README.md file
            readme_content = f'''# Neural Child Development Vault

This vault contains the development records of a neural child, including:
- Memories
- Emotional States
- Development Progress
- Language Learning
- Interactions

Created by the Neural Child Development System
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
            readme_path = self.vault_path / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
                
            self.logger.log_milestone(f"Initialized Obsidian vault at {self.vault_path}")
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'initialize_vault',
                'vault_path': str(self.vault_path)
            })
            raise
            
    def _write_markdown_file(self, folder: str, filename: str, frontmatter: Dict[str, Any], content: str):
        """Write a markdown file with YAML frontmatter."""
        try:
            folder_path = self.vault_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
            file_path = folder_path / f"{filename}.md"
            
            # Combine frontmatter and content
            file_content = "---\n"
            file_content += yaml.dump(frontmatter)
            file_content += "---\n\n"
            file_content += content
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
                
            self.logger.log_milestone(f"Wrote markdown file: {file_path}")
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': '_write_markdown_file',
                'file_path': str(file_path)
            })
            raise
    
    def store_memory(self, memory_data: Dict[str, Any]):
        """Store a memory in the vault."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        frontmatter = {
            "type": "memory",
            "created": datetime.now().isoformat(),
            "tags": ["memory", memory_data.get("type", "general")]
        }
        
        content = f"# Memory Entry: {timestamp}\n\n"
        content += f"## Content\n{memory_data.get('content', '')}\n\n"
        content += f"## Metadata\n```json\n{json.dumps(memory_data, indent=2)}\n```"
        
        self._write_markdown_file("Memories", f"memory_{timestamp}", frontmatter, content)
    
    def store_emotional_state(self, emotional_data: Dict[str, Any]):
        """Store an emotional state in the vault."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        frontmatter = {
            "type": "emotional_state",
            "created": datetime.now().isoformat(),
            "tags": ["emotional_state", emotional_data.get("primary_emotion", "unknown")]
        }
        
        content = f"# Emotional State: {timestamp}\n\n"
        content += f"## State\n{emotional_data.get('description', '')}\n\n"
        content += f"## Metadata\n```json\n{json.dumps(emotional_data, indent=2)}\n```"
        
        self._write_markdown_file("Emotional_States", f"emotional_{timestamp}", frontmatter, content)
    
    def store_development(self, development_data: Dict[str, Any]):
        """Store development information in the vault."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        frontmatter = {
            "type": "development",
            "created": datetime.now().isoformat(),
            "tags": ["development", development_data.get("stage", "unknown")]
        }
        
        content = f"# Development Entry: {timestamp}\n\n"
        content += f"## Progress\n{development_data.get('description', '')}\n\n"
        content += f"## Metadata\n```json\n{json.dumps(development_data, indent=2)}\n```"
        
        self._write_markdown_file("Development", f"development_{timestamp}", frontmatter, content)
    
    def store_language_learning(self, language_data: Dict[str, Any]):
        """Store language learning information in the vault."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        frontmatter = {
            "type": "language_learning",
            "created": datetime.now().isoformat(),
            "tags": ["language", language_data.get("category", "unknown")]
        }
        
        content = f"# Language Learning: {timestamp}\n\n"
        content += f"## New Knowledge\n{language_data.get('description', '')}\n\n"
        content += f"## Metadata\n```json\n{json.dumps(language_data, indent=2)}\n```"
        
        self._write_markdown_file("Language_Learning", f"language_{timestamp}", frontmatter, content)
    
    def store_interaction(self, interaction_data: Dict[str, Any]):
        """Store an interaction in the vault."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        frontmatter = {
            "type": "interaction",
            "created": datetime.now().isoformat(),
            "tags": ["interaction", interaction_data.get("type", "general")]
        }
        
        content = f"# Interaction: {timestamp}\n\n"
        content += f"## Dialog\n{interaction_data.get('content', '')}\n\n"
        content += f"## Metadata\n```json\n{json.dumps(interaction_data, indent=2)}\n```"
        
        self._write_markdown_file("Interactions", f"interaction_{timestamp}", frontmatter, content) 