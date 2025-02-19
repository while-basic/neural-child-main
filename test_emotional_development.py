# test_emotional_development.py
# Created by Christopher Celaya
# Test script for emotional development system

import torch
from emotional_development_system import EmotionalDevelopmentSystem, EmotionalCapability
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import math

class EmotionalScenario:
    def __init__(self):
        self.time = 0
        self.phase_duration = 100  # steps per phase
        self.novelty_cycle = 0
        self.comfort_cycle = 0
        self.social_cycle = 0
        
    def get_environment(self, step: int) -> dict:
        """Generate dynamic environmental conditions with cyclic variations"""
        self.time = step
        phase = (step // self.phase_duration) % 6
        
        # Update cycles with different frequencies
        self.novelty_cycle = math.sin(2 * math.pi * step / 150)  # Slower cycle
        self.comfort_cycle = math.sin(2 * math.pi * step / 200)  # Medium cycle
        self.social_cycle = math.sin(2 * math.pi * step / 100)   # Faster cycle
        
        if phase == 0:  # Peaceful baseline with subtle variations
            return {
                'safety': 0.8 + 0.1 * math.sin(step / 20),
                'novelty': 0.3 + 0.2 * self.novelty_cycle,
                'comfort': 0.7 + 0.1 * self.comfort_cycle,
                'stimulation': 0.4 + 0.2 * math.sin(step / 30)
            }
        elif phase == 1:  # Gradual stress introduction
            progress = (step % self.phase_duration) / self.phase_duration
            return {
                'safety': 0.8 - 0.4 * progress + 0.1 * math.sin(step / 15),
                'novelty': 0.3 + 0.4 * progress + 0.2 * self.novelty_cycle,
                'comfort': 0.7 - 0.3 * progress + 0.1 * self.comfort_cycle,
                'stimulation': 0.4 + 0.3 * progress + 0.1 * math.sin(step / 25)
            }
        elif phase == 2:  # High stress with fluctuations
            return {
                'safety': 0.4 + 0.2 * math.sin(step / 10),
                'novelty': 0.7 + 0.2 * self.novelty_cycle,
                'comfort': 0.4 + 0.2 * self.comfort_cycle,
                'stimulation': 0.7 + 0.2 * math.sin(step / 20)
            }
        elif phase == 3:  # Mother intervention with calming patterns
            return {
                'safety': 0.6 + 0.2 * math.sin(step / 30),
                'novelty': 0.5 + 0.1 * self.novelty_cycle,
                'comfort': 0.6 + 0.2 * self.comfort_cycle,
                'stimulation': 0.5 + 0.1 * math.sin(step / 40)
            }
        elif phase == 4:  # Recovery with increasing stability
            progress = (step % self.phase_duration) / self.phase_duration
            return {
                'safety': 0.6 + 0.2 * progress + 0.1 * math.sin(step / 25),
                'novelty': 0.5 - 0.2 * progress + 0.1 * self.novelty_cycle,
                'comfort': 0.6 + 0.2 * progress + 0.1 * self.comfort_cycle,
                'stimulation': 0.5 - 0.1 * progress + 0.1 * math.sin(step / 35)
            }
        else:  # New normal with gentle variations
            return {
                'safety': 0.75 + 0.1 * math.sin(step / 40),
                'novelty': 0.4 + 0.15 * self.novelty_cycle,
                'comfort': 0.7 + 0.1 * self.comfort_cycle,
                'stimulation': 0.45 + 0.15 * math.sin(step / 30)
            }
    
    def _noise(self, scale: float) -> float:
        return np.random.normal(0, scale)
    
    def get_mother_response(self, baby_state: dict, env_state: dict) -> dict:
        """Generate adaptive mother responses with enhanced emotional support"""
        # Base response with dynamic attunement
        response = {
            'joy': max(0.4, 0.5 + 0.3 * baby_state['joy'] + 0.1 * self.comfort_cycle),
            'trust': max(0.5, 0.6 + 0.3 * baby_state['trust'] + 0.1 * self.social_cycle),
            'fear': min(0.3, max(0.1, baby_state['fear'] * 0.5 - 0.1 * self.comfort_cycle)),
            'surprise': max(0.2, min(0.8, baby_state['surprise'] * 0.8 + 0.1 * self.novelty_cycle))
        }
        
        # Enhanced emotional support based on baby's state
        if baby_state['fear'] > 0.4:  # Moderate to high fear
            response['trust'] = min(1.0, response['trust'] + 0.3 + 0.1 * self.social_cycle)
            response['joy'] = min(1.0, response['joy'] + 0.2 + 0.1 * self.comfort_cycle)
            response['fear'] *= 0.4  # Stronger fear reduction
            
        if baby_state['surprise'] > 0.5:  # Moderate to high surprise
            response['trust'] = min(1.0, response['trust'] + 0.2 + 0.1 * self.social_cycle)
            response['joy'] = min(1.0, response['joy'] + 0.1 + 0.1 * self.comfort_cycle)
            response['surprise'] = max(0.2, response['surprise'] * 0.7)  # Maintain some surprise
            
        # Support secondary emotion development with cyclic influence
        if baby_state['curiosity'] > 0:  # Any curiosity
            response['joy'] = min(1.0, response['joy'] + 0.2 + 0.1 * self.novelty_cycle)
            response['surprise'] = min(0.8, response['surprise'] + 0.15 + 0.1 * self.novelty_cycle)
            response['trust'] = min(1.0, response['trust'] + 0.1 + 0.1 * self.social_cycle)
            
        if baby_state['frustration'] > 0.3:  # Moderate frustration
            response['trust'] = min(1.0, response['trust'] + 0.3 + 0.1 * self.social_cycle)
            response['joy'] = min(1.0, response['joy'] + 0.15 + 0.1 * self.comfort_cycle)
            response['fear'] *= 0.6  # Help reduce fear
            
        if baby_state['attachment'] > 0:  # Any attachment
            response['trust'] = min(1.0, response['trust'] + 0.25 + 0.15 * self.social_cycle)
            response['joy'] = min(1.0, response['joy'] + 0.15 + 0.1 * self.comfort_cycle)
            response['fear'] = max(0.1, response['fear'] * 0.7)
            
        if baby_state['pride'] > 0:  # Any pride
            response['joy'] = min(1.0, response['joy'] + 0.25 + 0.1 * self.novelty_cycle)
            response['trust'] = min(1.0, response['trust'] + 0.15 + 0.1 * self.social_cycle)
            response['surprise'] = min(0.7, response['surprise'] + 0.1)
            
        # Environmental adaptations with cyclic influence
        if env_state['safety'] < 0.5:  # Moderate to low safety
            response['trust'] = min(1.0, response['trust'] + 0.3 + 0.1 * self.social_cycle)
            response['joy'] = min(1.0, response['joy'] + 0.2 + 0.1 * self.comfort_cycle)
            response['fear'] = max(0.1, response['fear'] * 0.5)
            
        if env_state['novelty'] > 0.6:  # Moderate to high novelty
            response['surprise'] = min(0.8, response['surprise'] + 0.15 + 0.1 * self.novelty_cycle)
            response['trust'] = min(1.0, response['trust'] + 0.2 + 0.1 * self.social_cycle)
            response['joy'] = min(1.0, response['joy'] + 0.1 + 0.1 * self.comfort_cycle)
            
        if env_state['comfort'] < 0.4:  # Low comfort
            response['trust'] = min(1.0, response['trust'] + 0.25 + 0.1 * self.social_cycle)
            response['joy'] = min(1.0, response['joy'] + 0.15 + 0.1 * self.comfort_cycle)
            response['fear'] = max(0.1, response['fear'] * 0.6)
            
        # Add slight randomness for natural variation
        for emotion in response:
            variation = np.random.normal(0, 0.05)
            response[emotion] = max(0.1, min(1.0, response[emotion] + variation))
            
        # Ensure minimum emotional values for healthy interaction
        response = {k: max(v, 0.2) for k, v in response.items()}
            
        return response

def plot_emotional_development(history):
    """Plot comprehensive emotional development visualization"""
    plt.style.use('bmh')  # Using built-in style instead of seaborn
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Primary Emotions Over Time
    ax1 = plt.subplot(2, 2, 1)
    timestamps = range(len(history))
    emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
    colors = ['#FFD700', '#98FB98', '#FFB6C6', '#DDA0DD']
    
    for emotion, color in zip(emotions, colors):
        values = [entry['metrics'][emotion.lower()] for entry in history]
        ax1.plot(timestamps, values, label=emotion, color=color, linewidth=2)
        
    ax1.set_title('Primary Emotions Development', fontsize=12, pad=20)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Intensity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Secondary Emotions Over Time
    ax2 = plt.subplot(2, 2, 2)
    secondary_emotions = ['Curiosity', 'Frustration', 'Attachment', 'Pride']
    colors = ['#20B2AA', '#CD5C5C', '#4682B4', '#DAA520']
    
    for emotion, color in zip(secondary_emotions, colors):
        values = [entry['metrics'][emotion.lower()] for entry in history]
        ax2.plot(timestamps, values, label=emotion, color=color, linewidth=2)
        
    ax2.set_title('Secondary Emotions Development', fontsize=12, pad=20)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Intensity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Emotional Regulation Development
    ax3 = plt.subplot(2, 2, 3)
    regulation = [entry['metrics']['regulation_level'] for entry in history]
    stability = [entry.get('stability', 0) for entry in history]
    
    ax3.plot(timestamps, regulation, label='Regulation Strength', color='#4169E1', linewidth=2)
    ax3.plot(timestamps, stability, label='Emotional Stability', color='#32CD32', linewidth=2)
    ax3.set_title('Emotional Regulation Development', fontsize=12, pad=20)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Strength')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Environmental Response
    ax4 = plt.subplot(2, 2, 4)
    env_safety = [entry['environment'].get('safety', 0.5) for entry in history]  # Get safety level from dict
    trust_response = [entry['metrics']['trust'] for entry in history]
    
    ax4.plot(timestamps, env_safety, label='Environmental Safety', color='#20B2AA', linewidth=2)
    ax4.plot(timestamps, trust_response, label='Trust Response', color='#9370DB', linewidth=2)
    ax4.set_title('Environmental Response Analysis', fontsize=12, pad=20)
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emotional_development_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Initialize system and scenario
    device = torch.device('cpu')
    emotional_system = EmotionalDevelopmentSystem(device=device)
    scenario = EmotionalScenario()
    
    # Simulation parameters
    n_steps = 600  # 6 phases of 100 steps each
    assessment_interval = 50
    history = []
    
    print("\nStarting Enhanced Emotional Development Simulation...")
    print(f"Initial Capability: {emotional_system.capability.value}")
    
    try:
        for step in range(n_steps):
            # Generate environment and mother's response
            env_state = scenario.get_environment(step)
            current_state = emotional_system._format_emotional_response()
            mother_response = scenario.get_mother_response(current_state, env_state)
            
            # Process stimulus and update emotional state
            new_state = emotional_system.process_stimulus(mother_response, env_state)
            
            # Record state
            history.append({
                'timestamp': datetime.now(),
                'metrics': new_state,
                'environment': env_state,
                'mother_response': mother_response,
                'stability': emotional_system.assess_development()['stability']
            })
            
            # Periodic assessment and capability update
            if step % assessment_interval == 0:
                assessment = emotional_system.assess_development()
                emotional_system.update_capability(assessment)
                
                print(f"\nStep {step}")
                print(f"Capability Level: {emotional_system.capability.value}")
                print("Emotional State:")
                print(f"  Primary: Joy={new_state['joy']:.2f}, Trust={new_state['trust']:.2f}, "
                      f"Fear={new_state['fear']:.2f}, Surprise={new_state['surprise']:.2f}")
                print(f"  Secondary: Curiosity={new_state['curiosity']:.2f}, "
                      f"Frustration={new_state['frustration']:.2f}, "
                      f"Attachment={new_state['attachment']:.2f}, "
                      f"Pride={new_state['pride']:.2f}")
                print(f"Development Assessment: {assessment}")
                
                # Check for capability advancement
                if step > 0 and emotional_system.capability.value != history[-2]['metrics']['emotional_capability']:
                    print(f"\n🌟 Advanced to new capability: {emotional_system.capability.value}")
                    print(f"Assessment scores: {assessment}")
        
        # Plot final results
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