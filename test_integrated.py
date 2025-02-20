# test_integrated.py
# Description: Test and visualization script for integrated child development
# Created by: Christopher Celaya

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Any

from main import DigitalChild, DevelopmentalStage
from vision_development import VisionDevelopment
from development_logger import DevelopmentLogger

def plot_development_metrics(metrics: Dict[str, float], title: str) -> plt.Figure:
    """Create a radar plot of development metrics"""
    # Prepare the data
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    values += values[:1]
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add title
    plt.title(title, size=20, y=1.05)
    
    return fig

def plot_emotional_state(emotional_state: torch.Tensor, stage: str) -> plt.Figure:
    """Create a bar plot of emotional state"""
    # Prepare data
    emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
    values = emotional_state.cpu().numpy()
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=emotions, y=values, ax=ax)
    
    # Customize plot
    ax.set_ylim(0, 1)
    ax.set_title(f'Emotional State - {stage}', size=15)
    ax.set_ylabel('Intensity')
    
    return fig

def test_integrated_development():
    """Test and visualize integrated development system"""
    print("\n=== Testing Integrated Neural Child Development ===\n")
    
    # Create output directory for visualizations
    output_dir = Path("development_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize logger and child
    logger = DevelopmentLogger()
    child = DigitalChild(initial_stage=DevelopmentalStage.NEWBORN)
    child.logger = logger  # Set logger explicitly
    
    # Test image path - use a default tensor if image not available
    image_path = Path("test_images/faces.jpg")
    if not image_path.exists():
        print("\nNote: Test image not found. Using simulated visual input.")
        # Create a simple tensor representing visual input
        visual_input = torch.randn(3, 224, 224)  # RGB image tensor
    else:
        visual_input = image_path.as_posix()
    
    # Test different developmental stages
    stages = [
        DevelopmentalStage.NEWBORN,
        DevelopmentalStage.INFANT,
        DevelopmentalStage.EARLY_TODDLER,
        DevelopmentalStage.LATE_TODDLER,
        DevelopmentalStage.EARLY_PRESCHOOL,
        DevelopmentalStage.LATE_PRESCHOOL
    ]
    
    for stage in stages:
        print(f"\nTesting Stage: {stage.name}")
        print("=" * 50)
        
        # Set stage
        child.set_stage(stage, progress=0.5)  # 50% progress in each stage
        
        try:
            # Process integrated input
            result = child.process_integrated_input(
                visual_input=visual_input,
                context="Interacting with caregiver"
            )
            
            if result['success']:
                # Print response
                print("\nChild's Response:")
                print("-" * 20)
                print(result['response'])
                
                # Print metrics
                print("\nIntegrated Metrics:")
                print("-" * 20)
                for metric, value in result['integrated_metrics'].items():
                    print(f"{metric.replace('_', ' ').title():25}: {value:.2f}")
                
                # Create visualizations
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Plot integrated metrics
                fig_metrics = plot_development_metrics(
                    result['integrated_metrics'],
                    f"Integrated Development Metrics - {stage.name}"
                )
                fig_metrics.savefig(
                    output_dir / f"metrics_{stage.name}_{timestamp}.png"
                )
                
                # Plot emotional state
                fig_emotional = plot_emotional_state(
                    torch.tensor(result['emotional_state']),
                    stage.name
                )
                fig_emotional.savefig(
                    output_dir / f"emotional_{stage.name}_{timestamp}.png"
                )
                
                # Close figures
                plt.close(fig_metrics)
                plt.close(fig_emotional)
                
                print(f"\nVisualizations saved in {output_dir}")
                
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error processing stage {stage.name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-" * 50)

def create_development_summary(child: DigitalChild) -> plt.Figure:
    """Create a summary visualization of development progress"""
    # Get development data
    network_structure = child.get_network_structure()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Brain Network Metrics
    ax1 = plt.subplot(221)
    brain_metrics = network_structure['brain']['metrics']
    sns.barplot(x=list(brain_metrics.keys()), y=list(brain_metrics.values()), ax=ax1)
    ax1.set_title('Brain Network Metrics')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # 2. Emotional State
    ax2 = plt.subplot(222)
    emotional_state = network_structure['emotional']['state']
    sns.barplot(x=['Joy', 'Trust', 'Fear', 'Surprise'], y=emotional_state, ax=ax2)
    ax2.set_title('Emotional State')
    
    # 3. Development Metrics
    ax3 = plt.subplot(223)
    dev_metrics = network_structure['development']['metrics']
    sns.barplot(x=list(dev_metrics.keys()), y=list(dev_metrics.values()), ax=ax3)
    ax3.set_title('Development Metrics')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    
    # 4. Stage Progress
    ax4 = plt.subplot(224)
    ax4.text(0.5, 0.5, 
             f"Current Stage: {network_structure['development']['stage']}\n" +
             f"Age: {network_structure['development']['age_months']} months\n" +
             f"Language Stage: {network_structure['language']['stage']}\n" +
             f"Vocabulary Size: {network_structure['language']['vocabulary_size']}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax4.transAxes,
             fontsize=12)
    ax4.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Run integrated development test
    test_integrated_development()
    
    # Create and save development summary
    child = DigitalChild()
    summary_fig = create_development_summary(child)
    summary_fig.savefig("development_visualizations/development_summary.png")
    plt.close(summary_fig) 