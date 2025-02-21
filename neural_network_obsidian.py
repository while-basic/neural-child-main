# neural_network_obsidian.py
# Description: Neural network visualization system for Obsidian integration
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime
from visualization import NeuralNetworkVisualizer
from obsidian_api import ObsidianAPI
from obsidian_visualizer import ObsidianVisualizer

class NeuralNetworkObsidianVisualizer:
    """Neural network visualization system for Obsidian integration."""
    
    def __init__(self, vault_path: str):
        """Initialize the neural network Obsidian visualizer.
        
        Args:
            vault_path (str): Path to the Obsidian vault
        """
        self.vault_path = Path(vault_path)
        self.api = ObsidianAPI(vault_path=vault_path)
        self.visualizer = ObsidianVisualizer(vault_path)
        
    def create_network_note(self, model: nn.Module, name: str, description: str) -> None:
        """Create an Obsidian note for a neural network with visualizations.
        
        Args:
            model (nn.Module): The neural network model to visualize
            name (str): Name of the network
            description (str): Description of the network
        """
        # Create network directory if it doesn't exist
        network_dir = self.vault_path / "Network"
        network_dir.mkdir(parents=True, exist_ok=True)
        
        # Create attachments directory if it doesn't exist
        attachments_dir = network_dir / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        viz = NeuralNetworkVisualizer(model)
        
        # Create visualization paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch_path = attachments_dir / f"{name}_architecture_{timestamp}.png"
        weights_path = attachments_dir / f"{name}_weights_{timestamp}.png"
        activation_path = attachments_dir / f"{name}_activation_{timestamp}.png"
        
        # Generate visualizations
        viz.visualize_architecture(save_path=str(arch_path.with_suffix('')))
        viz.plot_weight_distributions(save_path=str(weights_path))
        
        # Generate sample activation heatmap
        sample_input = torch.randn(1, model.input_size) if hasattr(model, 'input_size') else torch.randn(1, 384)
        viz.create_activation_heatmap(sample_input, save_path=str(activation_path))
        
        # Create network metadata
        metadata = self._extract_network_metadata(model)
        
        # Create note content with proper escaping
        content = [
            f"# {name} Neural Network\n",
            f"## Description\n{description}\n",
            "## Architecture Visualization\n",
            f"![[attachments/{arch_path.name}]]\n",
            "## Weight Distributions\n",
            f"![[attachments/{weights_path.name}]]\n",
            "## Activation Heatmap\n",
            f"![[attachments/{activation_path.name}]]\n",
            "## Network Statistics\n",
            "```json\n",
            json.dumps(metadata, indent=2),
            "\n```\n",
            "## Mermaid Diagram\n",
            "```mermaid\n",
            "graph TD\n",
            self._create_mermaid_diagram(model),
            "\n```\n",
            "## Connections\n",
            "- [[Development/README|Development]] - Network evolution\n",
            "- [[Memories/README|Memories]] - Memory patterns\n",
            "- [[Language_Learning/README|Language Learning]] - Language processing\n",
            "\n## Recent Updates\n",
            "```dataview\n",
            'TABLE created as "Time", tags as "Tags"\n',
            f'FROM "#neural_network" and [[{name}]]\n',
            "SORT created DESC\n",
            "LIMIT 5\n",
            "```\n"
        ]
        
        # Save note
        note_path = network_dir / f"{name}.md"
        note_path.write_text("".join(content))
        
    def _extract_network_metadata(self, model: nn.Module) -> Dict[str, Any]:
        """Extract metadata about the neural network.
        
        Args:
            model (nn.Module): The neural network model
            
        Returns:
            Dict[str, Any]: Network metadata including layers and parameters
        """
        metadata = {
            "layers": [],
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "layer_sizes": []
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                metadata["layers"].append({
                    "name": name,
                    "type": "Linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features
                })
                metadata["layer_sizes"].append(module.out_features)
                
        return metadata
        
    def _create_mermaid_diagram(self, model: nn.Module) -> str:
        """Create a Mermaid diagram representation of the network.
        
        Args:
            model (nn.Module): The neural network model
            
        Returns:
            str: Mermaid diagram representation
        """
        diagram = []
        
        # Add nodes
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                clean_name = name.replace('.', '_')
                diagram.append(f"    {clean_name}[{name}<br/>{module.in_features}->{module.out_features}]")
                
        # Add connections
        prev_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                clean_name = name.replace('.', '_')
                if prev_name:
                    diagram.append(f"    {prev_name} --> {clean_name}")
                prev_name = clean_name
                
        return "\n".join(diagram)
        
    def visualize_all_networks(self, networks: Dict[str, nn.Module]) -> None:
        """Create visualizations for multiple networks.
        
        Args:
            networks (Dict[str, nn.Module]): Dictionary of networks to visualize
        """
        for name, network in networks.items():
            description = f"Neural network component: {name}"
            self.create_network_note(network, name, description)
            
    def create_network_index(self, networks: Dict[str, nn.Module]) -> None:
        """Create an index page for all neural networks.
        
        Args:
            networks (Dict[str, nn.Module]): Dictionary of networks to index
        """
        content = [
            "# Neural Network Overview\n",
            "\nThis document provides an overview of all neural networks in the system.\n",
            "\n## Network List\n"
        ]
        
        for name in networks.keys():
            content.append(f"- [[{name}]]\n")
            
        content.extend([
            "\n## Network Graph\n",
            "```mermaid\n",
            "graph TD\n"
        ])
        
        # Add connections between networks
        for name in networks.keys():
            content.append(f"    {name}[{name}]\n")
            
        # Add some basic connections
        content.extend([
            "    IntegratedBrain --> EmotionalNetwork\n",
            "    IntegratedBrain --> LanguageNetwork\n",
            "    IntegratedBrain --> MemoryNetwork\n",
            "    EmotionalNetwork --> DecisionNetwork\n",
            "    LanguageNetwork --> DecisionNetwork\n",
            "    MemoryNetwork --> DecisionNetwork\n",
            "```\n",
            "\n## Recent Changes\n",
            "```dataview\n",
            'TABLE file.mtime as "Last Modified"\n',
            'FROM "Network"\n',
            'WHERE file.name != "README" and file.name != "index"\n',
            "SORT file.mtime DESC\n",
            "```\n"
        ])
        
        # Save index
        index_path = self.vault_path / "Network" / "index.md"
        index_path.write_text("".join(content))