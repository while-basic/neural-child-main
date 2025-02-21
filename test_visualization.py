# test_visualization.py
# Description: Test script for visualization module
# Created by: Christopher Celaya

import torch
import numpy as np
import time
from visualization import (
    EmotionalStateVisualizer,
    NeuralNetworkVisualizer,
    plot_learning_metrics,
    plot_psychological_metrics
)
import os
from torch import nn
import sys
from pathlib import Path
import shutil

class SimpleNeuralNetwork(nn.Module):
    """A simple neural network for testing visualization"""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def test_emotional_visualizer():
    """Test emotional state visualization"""
    print("\nTesting Emotional State Visualizer...")
    
    # Create test directory
    os.makedirs("test_visualizations", exist_ok=True)
    
    # Initialize visualizer
    visualizer = EmotionalStateVisualizer()
    
    # Generate sample emotional states
    num_samples = 100
    time_points = np.linspace(0, 10, num_samples)
    
    for t in time_points:
        # Generate synthetic emotional state with some patterns
        joy = 0.5 + 0.3 * np.sin(t)
        trust = 0.6 + 0.2 * np.cos(t)
        fear = 0.3 + 0.1 * np.sin(2*t)
        surprise = 0.4 + 0.15 * np.cos(3*t)
        
        emotional_state = torch.tensor([joy, trust, fear, surprise])
        visualizer.add_emotional_state(emotional_state, t)
    
    # Test timeline plot
    print("Generating emotional timeline...")
    visualizer.plot_emotional_timeline("test_visualizations/emotional_timeline.png")
    
    # Test heatmap
    print("Generating emotional heatmap...")
    visualizer.create_emotional_heatmap("test_visualizations/emotional_heatmap.png")
    
    # Test 3D trajectory
    print("Generating 3D emotional trajectory...")
    visualizer.create_3d_emotional_trajectory("test_visualizations/emotional_trajectory.html")
    
    print("Emotional visualization tests completed!")

def test_neural_network_visualizer():
    """Test neural network visualization"""
    print("\nTesting Neural Network Visualizer...")
    
    # Initialize a simple neural network
    model = SimpleNeuralNetwork()
    visualizer = NeuralNetworkVisualizer(model)
    
    try:
        # Test architecture visualization
        print("Generating network architecture visualization...")
        visualizer.visualize_architecture(save_path="test_visualizations/network_architecture")
    except Exception as e:
        print("Warning: Could not generate network architecture visualization.")
        print("This might be because Graphviz is not installed on your system.")
        print("To install Graphviz:")
        print("1. Download from: https://graphviz.org/download/")
        print("2. Add to system PATH")
        print("3. Run: pip install graphviz")
    
    # Test weight distribution plots
    print("Generating weight distributions...")
    visualizer.plot_weight_distributions("test_visualizations/weight_distributions.png")
    
    # Test activation heatmap
    print("Generating activation heatmap...")
    test_input = torch.randn(1, 384)
    visualizer.create_activation_heatmap(test_input, "test_visualizations/activation_heatmap.png")
    
    print("Neural network visualization tests completed!")

def test_metrics_visualization():
    """Test learning metrics visualization"""
    print("\nTesting Metrics Visualization...")
    
    # Generate sample telemetry data
    telemetry = {
        'loss': [np.random.exponential(0.5) for _ in range(100)],
        'emotional_stability': [0.5 + 0.3 * np.sin(x/10) for x in range(100)],
        'conversation_quality': [min(1.0, 0.7 + 0.02 * x + np.random.normal(0, 0.1)) for x in range(100)],
        'psychological': {
            'attachment_style': [0.7 + 0.2 * np.random.random() for _ in range(10)],
            'defense_mechanisms': [0.6 + 0.3 * np.random.random() for _ in range(10)],
            'theory_of_mind': [0.5 + 0.4 * np.random.random() for _ in range(10)],
            'emotional_regulation': [0.8 + 0.1 * np.random.random() for _ in range(10)],
            'cognitive_development': [0.6 + 0.3 * np.random.random() for _ in range(10)],
            'social_bonds': [0.7 + 0.2 * np.random.random() for _ in range(10)],
            'moral_development': [0.5 + 0.4 * np.random.random() for _ in range(10)]
        }
    }
    
    # Test learning metrics plot
    print("Generating learning metrics visualization...")
    plot_learning_metrics(telemetry, "test_visualizations/learning_metrics.png")
    
    # Test psychological metrics plot
    print("Generating psychological metrics visualization...")
    plot_psychological_metrics(telemetry, "test_visualizations/psychological_metrics.html")
    
    print("Metrics visualization tests completed!")

def test_simple_network():
    """Test visualization with a simple network"""
    # Create a simple test network
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_size = 384  # Match the expected input size
            self.fc1 = nn.Linear(384, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x
    
    # Create test directory
    test_dir = Path("test_obsidian_vault")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    try:
        # Import visualization classes
        from neural_network_obsidian import NeuralNetworkObsidianVisualizer
        
        # Create visualizer
        visualizer = NeuralNetworkObsidianVisualizer(str(test_dir))
        
        # Create test network
        network = SimpleNetwork()
        
        # Test visualization
        print("Testing single network visualization...")
        visualizer.create_network_note(
            network,
            "TestNetwork",
            "A simple test network for visualization testing."
        )
        
        # Verify files were created
        network_dir = test_dir / "Network"
        attachments_dir = network_dir / "attachments"
        
        assert network_dir.exists(), "Network directory was not created"
        assert attachments_dir.exists(), "Attachments directory was not created"
        assert (network_dir / "TestNetwork.md").exists(), "Network note was not created"
        
        # Check for visualization files
        viz_files = list(attachments_dir.glob("TestNetwork_*.png"))
        assert len(viz_files) > 0, "No visualization files were created"
        
        print("Basic visualization test passed!")
        
        # Test index creation
        print("\nTesting index creation...")
        networks = {
            "TestNetwork1": network,
            "TestNetwork2": network
        }
        
        visualizer.create_network_index(networks)
        assert (network_dir / "index.md").exists(), "Index file was not created"
        
        print("Index creation test passed!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("\nTest directory cleaned up")

def main():
    """Run all visualization tests"""
    print("Starting visualization tests...")
    
    # Create test directory
    os.makedirs("test_visualizations", exist_ok=True)
    
    # Run tests
    test_emotional_visualizer()
    test_neural_network_visualizer()
    test_metrics_visualization()
    test_simple_network()
    
    print("\nAll visualization tests completed!")
    print("Test visualizations have been saved to the 'test_visualizations' directory.")
    print("\nPlease check the following files:")
    print("1. test_visualizations/emotional_timeline.png")
    print("2. test_visualizations/emotional_heatmap.png")
    print("3. test_visualizations/emotional_trajectory.html")
    print("4. test_visualizations/weight_distributions.png")
    print("5. test_visualizations/activation_heatmap.png")
    print("6. test_visualizations/learning_metrics.png")
    print("7. test_visualizations/psychological_metrics.html")
    print("\nNote: If network_architecture.png is missing, you need to install Graphviz.")

if __name__ == "__main__":
    main() 