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
                "Interactions",
                "Network"
            ]
            
            for folder in folders:
                folder_path = self.vault_path / folder
                folder_path.mkdir(parents=True, exist_ok=True)
                
            # Create .obsidian directory to mark this as an Obsidian vault
            obsidian_dir = self.vault_path / ".obsidian"
            obsidian_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a basic Obsidian config with graph settings
            config = {
                "basePath": str(self.vault_path),
                "useMarkdownLinks": True,
                "newLinkFormat": "relative",
                "attachmentFolderPath": "attachments",
                "graph": {
                    "zoomLevel": 1.0,
                    "showTags": True,
                    "showAttachments": False,
                    "hideUnresolved": False,
                    "showOrphans": True,
                    "showArrow": True,
                    "textFadeMultiplier": 0,
                    "nodeSizeMultiplier": 1,
                    "lineSizeMultiplier": 1,
                    "centerStrength": 0.518713248970312,
                    "repelStrength": 10,
                    "linkStrength": 1,
                    "linkDistance": 250,
                    "scale": 1
                }
            }
            
            config_path = obsidian_dir / "config"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            # Create a README.md file
            readme_content = f'''# Neural Child Development Vault

This vault contains the development records of a neural child, including:
- [[Memories/README|Memories]]
- [[Emotional_States/README|Emotional States]]
- [[Development/README|Development Progress]]
- [[Language_Learning/README|Language Learning]]
- [[Interactions/README|Interactions]]
- [[Network/README|Neural Network]]

Created by the Neural Child Development System
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
            readme_path = self.vault_path / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
                
            # Create README files for each section
            self._create_section_readmes()
                
            self.logger.log_milestone(f"Initialized Obsidian vault at {self.vault_path}")
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'initialize_vault',
                'vault_path': str(self.vault_path)
            })
            raise
            
    def _create_section_readmes(self):
        """Create README files for each section with network visualization info."""
        sections = {
            "Memories": "Contains episodic, semantic, and emotional memories formed by the neural child.",
            "Emotional_States": "Tracks emotional development and state changes over time.",
            "Development": "Records developmental milestones and stage progression.",
            "Language_Learning": "Documents vocabulary acquisition and language development.",
            "Interactions": "Stores conversation history and interaction patterns.",
            "Network": "Visualizes neural connections and relationship graphs."
        }
        
        for folder, description in sections.items():
            content = f'''# {folder}

{description}

## Network Visualization
This section is connected to:
{self._get_section_connections(folder)}

## Recent Updates
```dataview
TABLE created as "Time", tags as "Tags"
FROM "#{folder.lower()}"
SORT created DESC
LIMIT 5
```

## Network Graph
```dataview
TABLE WITHOUT ID
  file.link as "Entry",
  tags as "Tags",
  created as "Time"
FROM "#{folder.lower()}"
SORT created DESC
```
'''
            readme_path = self.vault_path / folder / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
    def _get_section_connections(self, section: str) -> str:
        """Get connection descriptions for a section."""
        connections = {
            "Memories": [
                "- [[Emotional_States/README|Emotional States]] - Memories have associated emotional context",
                "- [[Language_Learning/README|Language Learning]] - Semantic memories store learned concepts",
                "- [[Interactions/README|Interactions]] - Memories form during interactions"
            ],
            "Emotional_States": [
                "- [[Memories/README|Memories]] - Emotions influence memory formation",
                "- [[Development/README|Development]] - Emotional growth tracks development",
                "- [[Interactions/README|Interactions]] - Emotional responses to interactions"
            ],
            "Development": [
                "- [[Language_Learning/README|Language Learning]] - Language development stages",
                "- [[Emotional_States/README|Emotional States]] - Emotional development tracking",
                "- [[Network/README|Neural Network]] - Development affects network structure"
            ],
            "Language_Learning": [
                "- [[Memories/README|Memories]] - Words and concepts stored as memories",
                "- [[Development/README|Development]] - Language stages follow development",
                "- [[Interactions/README|Interactions]] - Learning through conversation"
            ],
            "Interactions": [
                "- [[Memories/README|Memories]] - Interactions create memories",
                "- [[Emotional_States/README|Emotional States]] - Emotional responses",
                "- [[Language_Learning/README|Language Learning]] - Learning from interaction"
            ],
            "Network": [
                "- [[Development/README|Development]] - Network structure evolution",
                "- [[Memories/README|Memories]] - Memory connection patterns",
                "- [[Language_Learning/README|Language Learning]] - Language network formation"
            ]
        }
        return "\n".join(connections.get(section, []))
    
    def _write_markdown_file(self, folder: str, filename: str, frontmatter: Dict[str, Any], content: str):
        """Write a markdown file with YAML frontmatter and network connections."""
        try:
            folder_path = self.vault_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
            file_path = folder_path / f"{filename}.md"
            
            # Add network metadata to frontmatter
            frontmatter.update({
                "connections": self._get_entry_connections(folder, frontmatter),
                "graph": {
                    "color": self._get_node_color(folder),
                    "size": self._get_node_size(frontmatter)
                }
            })
            
            # Add network visualization section to content
            content += "\n\n## Connections\n"
            content += self._format_connections(frontmatter["connections"])
            
            # Add dataview queries
            content += "\n\n## Related Entries\n"
            content += "```dataview\n"
            content += "TABLE WITHOUT ID file.link as Entry, type as Type, tags as Tags\n"
            content += f'FROM #{folder.lower()} and !"{file_path.name}"\n'
            content += "WHERE contains(connections, this.file.name)\n"
            content += "SORT created DESC\n"
            content += "```\n"
            
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
            
    def _get_entry_connections(self, folder: str, frontmatter: Dict[str, Any]) -> List[str]:
        """Get list of connected entries based on entry type and metadata."""
        connections = []
        entry_type = frontmatter.get("type", "")
        
        if entry_type == "language_learning":
            # Connect to development stage entries
            connections.append(f"[[Development/README|Current Stage: {frontmatter.get('stage', 'UNKNOWN')}]]")
            
        elif entry_type == "emotional_state":
            # Connect to related emotional memories
            connections.append("[[Memories/README|Related Emotional Memories]]")
            
        elif entry_type == "development":
            # Connect to previous stage
            prev_stage = frontmatter.get("previous_stage")
            if prev_stage:
                connections.append(f"[[Development/README|Previous Stage: {prev_stage}]]")
                
        elif entry_type == "interaction":
            # Connect to emotional state and learning
            if "emotions" in frontmatter:
                connections.append("[[Emotional_States/README|Emotional Response]]")
            if "learning" in frontmatter:
                connections.append("[[Language_Learning/README|Learning Outcomes]]")
                
        # Add connection to network visualization
        connections.append("[[Network/README|Neural Network]]")
        
        return connections
    
    def _get_node_color(self, folder: str) -> str:
        """Get node color for graph visualization."""
        colors = {
            "Memories": "#ff6b6b",  # Red
            "Emotional_States": "#4ecdc4",  # Teal
            "Development": "#95a5a6",  # Gray
            "Language_Learning": "#f1c40f",  # Yellow
            "Interactions": "#3498db",  # Blue
            "Network": "#2ecc71"  # Green
        }
        return colors.get(folder, "#bdc3c7")  # Default to light gray
    
    def _get_node_size(self, frontmatter: Dict[str, Any]) -> float:
        """Get node size based on entry importance."""
        if "emotional_value" in frontmatter:
            return 1.0 + float(frontmatter["emotional_value"])
        elif "stage" in frontmatter:
            return 2.0  # Development entries are larger
        return 1.0  # Default size
    
    def _format_connections(self, connections: List[str]) -> str:
        """Format connections list as markdown."""
        if not connections:
            return "No direct connections."
        return "\n".join([f"- {conn}" for conn in connections])

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