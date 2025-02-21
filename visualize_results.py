# visualize_results.py
# Description: Visualization script for neural child development results
# Created by: Christopher Celaya

import matplotlib.pyplot as plt
import numpy as np
from main import IntegratedBrain, DigitalChild
from developmental_stages import DevelopmentalStage
import seaborn as sns

def visualize_developmental_stages():
    """Visualize metrics across different developmental stages."""
    # Create brain instances for different stages
    stages = [
        DevelopmentalStage.NEWBORN,
        DevelopmentalStage.INFANT,
        DevelopmentalStage.EARLY_TODDLER
    ]
    
    # Initialize data collection
    integration_levels = []
    memory_scores = []
    emotional_scores = []
    learning_rates = []
    
    # Collect data for each stage
    for stage in stages:
        child = DigitalChild(stage=stage)
        brain = child.brain
        
        # Process dummy input to get metrics
        visual_input = torch.randn(1, brain.input_dim).to(brain.device)
        auditory_input = torch.randn(1, brain.input_dim).to(brain.device)
        emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=brain.device)
        
        output = brain(visual_input, auditory_input, emotions)
        
        # Collect metrics
        integration_levels.append(torch.mean(output['integrated']).item())
        memory_scores.append(torch.mean(output['memory']['integrated_memory']).item())
        emotional_scores.append(torch.mean(output['emotional']['emotions']).item())
        learning_rates.append(brain.developmental_factors['learning'])
    
    # Create visualization
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Neural Child Development Metrics Across Stages', fontsize=16, y=0.95)
    
    # Plot integration levels
    stage_names = [stage.name for stage in stages]
    
    # Integration levels
    sns.barplot(x=stage_names, y=integration_levels, ax=ax1, palette='viridis')
    ax1.set_title('Integration Levels')
    ax1.set_ylim(0, 1)
    
    # Memory scores
    sns.barplot(x=stage_names, y=memory_scores, ax=ax2, palette='magma')
    ax2.set_title('Memory Processing')
    ax2.set_ylim(0, 1)
    
    # Emotional scores
    sns.barplot(x=stage_names, y=emotional_scores, ax=ax3, palette='plasma')
    ax3.set_title('Emotional Processing')
    ax3.set_ylim(0, 1)
    
    # Learning rates
    sns.barplot(x=stage_names, y=learning_rates, ax=ax4, palette='inferno')
    ax4.set_title('Learning Rates')
    ax4.set_ylim(0, 1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('development_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_reward_system():
    """Visualize the reward system components."""
    # Create test scenarios
    scenarios = [
        ("Positive Interaction", "I'm really excited!", "That's wonderful, I'm excited too!"),
        ("Emotional Misalignment", "I'm feeling sad", "Let's celebrate!"),
        ("Memory Recall", "What's my favorite color?", "Your favorite color is blue!"),
        ("Curiosity", "Tell me about space", "Space is fascinating! What interests you most?")
    ]
    
    # Initialize brain
    brain = IntegratedBrain()
    
    # Collect rewards
    rewards = []
    components = {
        'emotional': [],
        'memory': [],
        'curiosity': [],
        'flow': []
    }
    
    for _, user_input, ai_response in scenarios:
        # Get overall reward
        reward = brain.get_reward(user_input, ai_response)
        rewards.append(reward)
        
        # Get component scores
        components['emotional'].append(
            1.0 - abs(brain._analyze_sentiment(user_input) - brain._analyze_sentiment(ai_response))
        )
        components['memory'].append(brain._evaluate_memory_recall(ai_response))
        components['curiosity'].append(brain._evaluate_curiosity(ai_response))
        components['flow'].append(brain._evaluate_conversation_flow(user_input, ai_response))
    
    # Create visualization
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))
    
    # Plot stacked bars for components
    bottom = np.zeros(len(scenarios))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    for (component, values), color in zip(components.items(), colors):
        plt.bar([s[0] for s in scenarios], values, bottom=bottom, label=component.capitalize(),
                color=color, alpha=0.7)
        bottom += np.array(values)
    
    # Plot total rewards
    plt.plot([s[0] for s in scenarios], rewards, 'ko-', label='Total Reward', linewidth=2)
    
    plt.title('Reward System Analysis', fontsize=16)
    plt.xlabel('Interaction Type')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('reward_system.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import torch
    print("Generating visualizations...")
    visualize_developmental_stages()
    visualize_reward_system()
    print("Visualizations saved as 'development_metrics.png' and 'reward_system.png'") 