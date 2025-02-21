# visualize_networks.py
# Description: Script to visualize neural networks in Obsidian
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from pathlib import Path
import shutil
import os

def setup_obsidian_vault():
    vault_path = Path("obsidian_vault")
    network_dir = vault_path / "Network"
    attachments_dir = network_dir / "attachments"
    
    # Create directories
    network_dir.mkdir(parents=True, exist_ok=True)
    attachments_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README if it doesn't exist
    readme_path = network_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# Neural Network Documentation

This directory contains documentation and visualizations for all neural networks in the system.

## Structure
- `attachments/` - Contains all network visualizations
- Individual `.md` files for each network
- `index.md` - Overview of all networks

## Visualization Types
1. Architecture diagrams
2. Weight distributions
3. Activation heatmaps
4. Network statistics
5. Mermaid diagrams showing connections

## Usage
Open this vault in Obsidian to view the interactive documentation.
"""
        readme_path.write_text(readme_content)
    
    return vault_path

def main():
    try:
        # Set up vault structure
        print("Setting up Obsidian vault structure...")
        vault_path = setup_obsidian_vault()
        
        # Initialize the visualizer
        print("Initializing visualizer...")
        from neural_network_obsidian import NeuralNetworkObsidianVisualizer
        visualizer = NeuralNetworkObsidianVisualizer(str(vault_path))
        
        # Create our neural networks
        print("Creating neural networks...")
        networks = {
            "IntegratedBrain": nn.Sequential(
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            ),
            "EmotionalNetwork": nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 12)  # 12 emotional dimensions
            ),
            "LanguageNetwork": nn.Sequential(
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            ),
            "MemoryNetwork": nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            ),
            "DecisionNetwork": nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        }
        
        # Add input_size attribute to each network for visualization
        input_sizes = {
            "IntegratedBrain": 384,
            "EmotionalNetwork": 256,
            "LanguageNetwork": 384,
            "MemoryNetwork": 256,
            "DecisionNetwork": 256
        }
        
        for name, network in networks.items():
            network.input_size = input_sizes[name]
        
        # Create descriptions for each network
        descriptions = {
            "IntegratedBrain": "The main brain architecture that integrates all neural components for cognitive processing.",
            "EmotionalNetwork": "Processes and regulates emotional states and responses.",
            "LanguageNetwork": "Handles language understanding, generation, and development.",
            "MemoryNetwork": "Manages memory formation, storage, and retrieval.",
            "DecisionNetwork": "Makes decisions based on integrated information from other networks."
        }
        
        # Visualize each network
        print("\nCreating network visualizations...")
        for name, network in networks.items():
            print(f"Visualizing {name}...")
            visualizer.create_network_note(network, name, descriptions[name])
        
        # Create index page
        print("\nCreating network index...")
        visualizer.create_network_index(networks)
        
        print("\nDone! Neural network visualizations have been created in the Obsidian vault.")
        print(f"Vault location: {vault_path.absolute()}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 