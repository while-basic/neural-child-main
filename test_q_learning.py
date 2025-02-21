"""
Test Script for Q-Learning Implementation
Created by: Christopher Celaya

This script tests the Q-Learning implementation in the neural child development system.
"""

import torch
import matplotlib.pyplot as plt
from main import DigitalChild
from integrated_brain import DevelopmentalStage
import numpy as np
from typing import Dict, List
import seaborn as sns

def plot_results(rewards_history: List[float], 
                emotion_history: List[Dict[str, float]], 
                learning_metrics: List[Dict[str, float]]):
    """Plot the results of the training"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Rewards over time
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.title('Neural Child Development Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot 2: Emotional trends
    plt.subplot(2, 2, 2)
    emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust']
    for emotion in emotions:
        values = [state[emotion] for state in emotion_history]
        plt.plot(values, label=emotion)
    plt.title('Emotional Development')
    plt.xlabel('Episode')
    plt.ylabel('Emotion Intensity')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Learning metrics
    plt.subplot(2, 2, 3)
    metrics = ['vocabulary_size', 'grammar_complexity', 'comprehension_level']
    for metric in metrics:
        values = [m[metric] for m in learning_metrics]
        plt.plot(values, label=metric)
    plt.title('Learning Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Heatmap of emotional correlations
    plt.subplot(2, 2, 4)
    emotion_data = np.array([[state[e] for e in emotions] for state in emotion_history])
    correlation_matrix = np.corrcoef(emotion_data.T)
    sns.heatmap(correlation_matrix, 
                xticklabels=emotions, 
                yticklabels=emotions, 
                cmap='coolwarm', 
                center=0,
                annot=True,
                fmt='.2f')
    plt.title('Emotional Correlations')
    
    plt.tight_layout()
    plt.savefig('neural_child_results.png')
    plt.close()

def test_q_learning():
    """Test the Q-Learning implementation with a digital child"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing Digital Child...")
    # Initialize digital child
    child = DigitalChild()
    
    # Training parameters
    num_episodes = 1000  # Increased from 100
    max_steps = 500     # Increased from 50
    rewards_history = []
    emotion_history = []
    learning_metrics = []
    
    # Test inputs with increasing complexity
    test_inputs = [
        "I love you",
        "I hate you",
        "You're beautiful",
        "You're ugly",
        "The world is ending",
        "Life is wonderful",
        "I'm so angry at you",
        "You make me happy",
        "I don't trust you",
        "You're my best friend"
    ]
    
    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        episode_reward = 0
        
        for step in range(max_steps):
            # Generate input with increasing complexity
            complexity = min(1.0, (episode / num_episodes) + (step / max_steps))
            input_idx = int((len(test_inputs) - 1) * complexity)
            input_text = test_inputs[input_idx]
            
            input_data = {
                'text': input_text,
                'emotion': 'mixed',
                'context': 'emotional_learning',
                'complexity': complexity
            }
            
            # Interact with child
            result = child.interact(input_data)
            episode_reward += result['reward']
            
            # Store emotional state
            if 'brain_state' in result:
                brain_state = result['brain_state']
                if isinstance(brain_state, dict):
                    emotion_state = {
                        'joy': brain_state.get('emotional_valence', 0.5),
                        'sadness': 1.0 - brain_state.get('emotional_valence', 0.5),
                        'anger': brain_state.get('arousal', 0.5),
                        'fear': brain_state.get('stress', 0.5),
                        'surprise': brain_state.get('consciousness', 0.5),
                        'trust': 1.0 - brain_state.get('fatigue', 0.5)
                    }
                else:
                    # Default emotional state if brain_state is not a dict
                    emotion_state = {
                        'joy': 0.5,
                        'sadness': 0.5,
                        'anger': 0.5,
                        'fear': 0.5,
                        'surprise': 0.5,
                        'trust': 0.5
                    }
                emotion_history.append(emotion_state)
            
            # Store learning metrics
            if hasattr(child.brain, 'language_development'):
                metrics = child.brain.language_development.get_metrics()
                learning_metrics.append(metrics)
            
            # Print progress every 50 episodes
            if episode % 50 == 0 and step == 0:
                print(f"\nEpisode {episode}/{num_episodes}")
                print(f"Current stage: {result['developmental_stage']}")
                print(f"Stage progress: {result['stage_progress']:.2f}")
                print(f"Average reward: {result['metrics']['avg_reward']:.2f}")
                if 'level_up' in result and result['level_up']:
                    print("\n🎉 Level Up!")
                    print(f"Achievements: {result['level_up']['achievements']}")
                print("---")
        
        rewards_history.append(episode_reward)
    
    print("\nTraining completed!")
    print(f"Final stage: {result['developmental_stage']}")
    print(f"Final stage progress: {result['stage_progress']:.2f}")
    
    # Plot results
    plot_results(rewards_history, emotion_history, learning_metrics)
    print("\nResults saved to neural_child_results.png")

if __name__ == "__main__":
    test_q_learning() 