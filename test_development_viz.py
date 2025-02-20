# test_development_viz.py
# Description: Test script for visualizing emotional development and monitoring progress
# Created by: Christopher Celaya

import torch
from main import DigitalChild
from visualization import EmotionalStateVisualizer, plot_psychological_metrics
import os
from datetime import datetime, timedelta
import numpy as np
import time

def test_emotional_development():
    """Test and visualize emotional development over time"""
    print("\nTesting Emotional Development Visualization...")
    
    # Create directories for results
    os.makedirs("development_results", exist_ok=True)
    os.makedirs("development_results/visualizations", exist_ok=True)
    
    # Initialize child and visualizer
    child = DigitalChild()
    visualizer = EmotionalStateVisualizer()
    
    # Simulate development over time (24 hours compressed into a few minutes)
    start_time = datetime.now()
    print("\nSimulating 24 hours of emotional development...")
    
    # Test scenarios to trigger emotional responses
    scenarios = [
        "Playing with toys",
        "Meeting new people",
        "Hearing loud noises",
        "Getting hugged",
        "Learning new words",
        "Taking a nap",
        "Eating favorite food",
        "Watching butterflies",
        "Thunder and lightning",
        "Bedtime stories"
    ]
    
    telemetry = {
        'psychological': {
            'attachment_style': [],
            'defense_mechanisms': [],
            'theory_of_mind': [],
            'emotional_regulation': [],
            'cognitive_development': [],
            'social_bonds': [],
            'moral_development': []
        }
    }
    
    try:
        for hour in range(24):
            # Simulate time passing
            current_time = start_time + timedelta(hours=hour)
            
            # Select random scenario
            scenario = np.random.choice(scenarios)
            print(f"\nHour {hour}: {scenario}")
            
            # Update emotional state based on scenario
            if "loud" in scenario or "thunder" in scenario:
                # Increase fear and surprise
                delta = torch.tensor([0.1, -0.1, 0.7, 0.6], device='cuda')
            elif "hug" in scenario or "play" in scenario:
                # Increase joy and trust
                delta = torch.tensor([0.8, 0.7, -0.2, 0.3], device='cuda')
            elif "learn" in scenario:
                # Increase trust and surprise
                delta = torch.tensor([0.4, 0.6, 0.1, 0.7], device='cuda')
            else:
                # Random small changes
                delta = torch.randn(4, device='cuda') * 0.3
            
            # Update child's emotional state
            child.emotional_state = torch.clamp(child.emotional_state + delta, 0, 1)
            
            # Add to visualizer
            visualizer.add_emotional_state(child.emotional_state, current_time.timestamp())
            
            # Update psychological metrics
            for metric in telemetry['psychological']:
                # Simulate development with some randomness
                value = min(1.0, 0.3 + (hour/24.0) + np.random.normal(0, 0.1))
                telemetry['psychological'][metric].append(value)
            
            # Display current state
            print(f"Emotional State: {child.express_feeling()}")
            print(f"Joy: {child.emotional_state[0]:.2f}, Trust: {child.emotional_state[1]:.2f}")
            print(f"Fear: {child.emotional_state[2]:.2f}, Surprise: {child.emotional_state[3]:.2f}")
            
            # Small delay to simulate time passing
            time.sleep(0.5)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Emotional timeline
        visualizer.plot_emotional_timeline("development_results/visualizations/emotional_timeline.png")
        print("Generated emotional timeline")
        
        # Emotional heatmap
        visualizer.create_emotional_heatmap("development_results/visualizations/emotional_heatmap.png")
        print("Generated emotional heatmap")
        
        # 3D emotional trajectory
        visualizer.create_3d_emotional_trajectory("development_results/visualizations/emotional_trajectory.html")
        print("Generated 3D emotional trajectory")
        
        # Psychological metrics
        plot_psychological_metrics(telemetry, "development_results/visualizations/psychological_metrics.html")
        print("Generated psychological metrics visualization")
        
        print("\nVisualizations have been saved to development_results/visualizations/")
        print("Please check the following files:")
        print("1. emotional_timeline.png - Shows emotional state changes over time")
        print("2. emotional_heatmap.png - Shows correlations between emotions")
        print("3. emotional_trajectory.html - Shows 3D visualization of emotional development")
        print("4. psychological_metrics.html - Shows overall developmental progress")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    test_emotional_development() 