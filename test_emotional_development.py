# test_emotional_development.py
# Created by Christopher Celaya
# Test script for emotional development system

import torch
from emotional_development_system import EmotionalDevelopmentSystem, EmotionalCapability
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

def simulate_environmental_changes():
    """Simulate changing environmental conditions"""
    base_safety = 0.7 + np.random.normal(0, 0.1)
    base_novelty = 0.4 + np.sin(time.time() * 0.1) * 0.3
    base_comfort = 0.6 + np.random.normal(0, 0.1)
    base_stimulation = 0.5 + np.cos(time.time() * 0.1) * 0.3
    
    return {
        'safety': max(0, min(1, base_safety)),
        'novelty': max(0, min(1, base_novelty)),
        'comfort': max(0, min(1, base_comfort)),
        'stimulation': max(0, min(1, base_stimulation))
    }

def simulate_mother_response(baby_state, env_state):
    """Simulate an attuned mother's response"""
    # Base response with high attunement
    response = {
        'joy': 0.6 + 0.2 * baby_state['joy'],
        'trust': 0.7 + 0.2 * baby_state['trust'],
        'fear': max(0.1, baby_state['fear'] * 0.5),  # Regulated fear response
        'surprise': baby_state['surprise'] * 0.7
    }
    
    # Adjust based on environment
    if env_state['safety'] < 0.4:
        response['trust'] += 0.2  # Increased reassurance
    if env_state['novelty'] > 0.7:
        response['surprise'] += 0.1  # Shared interest
    
    # Ensure values are in valid range
    return {k: max(0, min(1, v)) for k, v in response.items()}

def plot_emotional_development(history):
    """Plot emotional development over time"""
    timestamps = [entry['timestamp'] for entry in history]
    emotions = {
        'joy': [entry['metrics']['joy'] for entry in history],
        'trust': [entry['metrics']['trust'] for entry in history],
        'fear': [entry['metrics']['fear'] for entry in history],
        'surprise': [entry['metrics']['surprise'] for entry in history]
    }
    
    plt.figure(figsize=(12, 6))
    for emotion, values in emotions.items():
        plt.plot(range(len(values)), values, label=emotion)
    
    plt.title('Emotional Development Over Time')
    plt.xlabel('Interaction Steps')
    plt.ylabel('Emotion Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Initialize system
    device = torch.device('cpu')
    emotional_system = EmotionalDevelopmentSystem(device=device)
    
    # Simulation parameters
    n_steps = 1000
    assessment_interval = 50
    history = []
    
    print("Starting Emotional Development Simulation...")
    print(f"Initial Capability: {emotional_system.capability.value}")
    
    try:
        for step in range(n_steps):
            # Simulate environment
            env_state = simulate_environmental_changes()
            
            # Get current emotional state
            current_state = emotional_system._format_emotional_response()
            
            # Get mother's response
            mother_response = simulate_mother_response(current_state, env_state)
            
            # Process stimulus and update emotional state
            new_state = emotional_system.process_stimulus(mother_response, env_state)
            
            # Record state
            history.append({
                'timestamp': datetime.now(),
                'metrics': new_state,
                'environment': env_state,
                'mother_response': mother_response
            })
            
            # Periodic assessment
            if step % assessment_interval == 0:
                assessment = emotional_system.assess_development()
                emotional_system.update_capability(assessment)
                
                print(f"\nStep {step}")
                print(f"Capability Level: {emotional_system.capability.value}")
                print(f"Emotional State: {new_state}")
                print(f"Development Assessment: {assessment}")
                
                # Check for capability advancement
                if step > 0 and emotional_system.capability.value != history[-1]['metrics']['emotional_capability']:
                    print(f"\n🌟 Advanced to new capability: {emotional_system.capability.value}")
                    print(f"Assessment scores: {assessment}")
        
        # Plot results
        plot_emotional_development(history)
        
        # Final assessment
        final_assessment = emotional_system.assess_development()
        print("\nFinal Development Assessment:")
        print(f"Emotional Capability: {emotional_system.capability.value}")
        print(f"Stability: {final_assessment['stability']:.2f}")
        print(f"Regulation: {final_assessment['regulation']:.2f}")
        print(f"Attachment: {final_assessment['attachment']:.2f}")
        print(f"Complexity: {final_assessment['complexity']:.2f}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        # Still show results
        plot_emotional_development(history)

if __name__ == "__main__":
    main() 