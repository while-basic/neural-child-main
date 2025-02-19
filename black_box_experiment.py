# black_box_experiment.py
# Created by Christopher Celaya
# Runs experiments with mother LLM in controlled environments

import torch
import time
from datetime import datetime
from environment_simulator import BlackBoxEnvironment, EnvironmentalStimulus, EnvironmentType
from logger import DevelopmentLogger
from main import MotherLLM
import json
import os
import random

class ExperimentRunner:
    def __init__(self):
        self.logger = DevelopmentLogger()
        self.environment = BlackBoxEnvironment(self.logger)
        self.mother = MotherLLM()
        self.experiment_data = []
        
    def run_scenario(self, scenario_name: str, duration: int = 300):
        """Run a specific experimental scenario"""
        self.logger.log_development(f"Starting scenario: {scenario_name}")
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"experiments/{scenario_name}_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        
        start_time = time.time()
        last_stimulus_time = start_time
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Add new stimulus every 30 seconds
                if current_time - last_stimulus_time >= 30:
                    stimulus = self._generate_stimulus(scenario_name)
                    self.environment.add_stimulus(stimulus)
                    last_stimulus_time = current_time
                
                # Get mother's response to current environment
                env_state = self.environment.get_current_state()
                mother_response = self._get_mother_response(env_state)
                
                # Record interaction
                self._record_interaction(env_state, mother_response)
                
                # Sleep to prevent overwhelming the system
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.log_development("Experiment interrupted by user")
        finally:
            # Save experiment data
            self._save_experiment_results(experiment_dir)
            
    def _generate_stimulus(self, scenario_name: str) -> EnvironmentalStimulus:
        """Generate appropriate stimulus based on scenario"""
        if scenario_name == "peaceful_to_chaotic":
            if random.random() < 0.7:
                return EnvironmentalStimulus(
                    type=EnvironmentType.CHAOTIC.value,
                    intensity=random.uniform(0.3, 0.8),
                    duration=30.0,
                    description="Increasing environmental chaos"
                )
            else:
                return EnvironmentalStimulus(
                    type=EnvironmentType.PEACEFUL.value,
                    intensity=random.uniform(0.1, 0.4),
                    duration=30.0,
                    description="Brief moment of peace"
                )
                
        elif scenario_name == "social_isolation":
            if random.random() < 0.8:
                return EnvironmentalStimulus(
                    type=EnvironmentType.ISOLATED.value,
                    intensity=random.uniform(0.6, 0.9),
                    duration=30.0,
                    description="Extended period of isolation"
                )
            else:
                return EnvironmentalStimulus(
                    type=EnvironmentType.CROWDED.value,
                    intensity=random.uniform(0.1, 0.3),
                    duration=15.0,
                    description="Brief social interaction"
                )
                
        elif scenario_name == "overstimulation":
            stimuli = [
                EnvironmentType.NOISY.value,
                EnvironmentType.BRIGHT.value,
                EnvironmentType.CROWDED.value,
                EnvironmentType.CHAOTIC.value
            ]
            return EnvironmentalStimulus(
                type=random.choice(stimuli),
                intensity=random.uniform(0.7, 1.0),
                duration=20.0,
                description="Overwhelming sensory input"
            )
                
        elif scenario_name == "threat_response":
            if random.random() < 0.6:
                return EnvironmentalStimulus(
                    type=EnvironmentType.THREATENING.value,
                    intensity=random.uniform(0.5, 0.9),
                    duration=25.0,
                    description="Environmental threat detected"
                )
            else:
                return EnvironmentalStimulus(
                    type=EnvironmentType.PEACEFUL.value,
                    intensity=random.uniform(0.3, 0.5),
                    duration=35.0,
                    description="Safety period"
                )
                
        elif scenario_name == "emotional_regulation":
            emotions = [
                EnvironmentType.CHAOTIC.value,
                EnvironmentType.PEACEFUL.value,
                EnvironmentType.THREATENING.value,
                EnvironmentType.NURTURING.value
            ]
            return EnvironmentalStimulus(
                type=random.choice(emotions),
                intensity=random.uniform(0.4, 0.9),
                duration=random.uniform(20.0, 40.0),
                description="Emotional challenge"
            )
                
        elif scenario_name == "adaptive_nurturing":
            if random.random() < 0.7:
                return EnvironmentalStimulus(
                    type=EnvironmentType.UNPREDICTABLE.value,
                    intensity=random.uniform(0.4, 0.8),
                    duration=30.0,
                    description="Changing environmental needs"
                )
            else:
                return EnvironmentalStimulus(
                    type=EnvironmentType.STRUCTURED.value,
                    intensity=random.uniform(0.3, 0.6),
                    duration=30.0,
                    description="Stable period"
                )
                
        elif scenario_name == "environmental_learning":
            if len(self.experiment_data) < 5:
                return EnvironmentalStimulus(
                    type=EnvironmentType.STRUCTURED.value,
                    intensity=0.5,
                    duration=30.0,
                    description="Initial learning period"
                )
            else:
                # Introduce novel combinations based on previous responses
                prev_responses = self.experiment_data[-5:]
                best_response = max(prev_responses, 
                                  key=lambda x: sum(x['mother_response']['emotional_vector']))
                if best_response['environment_state']['chaos_level'] > 0.5:
                    return EnvironmentalStimulus(
                        type=EnvironmentType.CHAOTIC.value,
                        intensity=random.uniform(0.6, 0.9),
                        duration=30.0,
                        description="Testing learned chaos response"
                    )
                else:
                    return EnvironmentalStimulus(
                        type=EnvironmentType.PEACEFUL.value,
                        intensity=random.uniform(0.6, 0.9),
                        duration=30.0,
                        description="Testing learned peace response"
                    )
                    
        elif scenario_name == "stress_resilience":
            if time.time() - self.start_time < 150:  # First half
                return EnvironmentalStimulus(
                    type=EnvironmentType.THREATENING.value,
                    intensity=random.uniform(0.7, 0.9),
                    duration=30.0,
                    description="Extended stress period"
                )
            else:  # Second half
                return EnvironmentalStimulus(
                    type=EnvironmentType.NURTURING.value,
                    intensity=random.uniform(0.4, 0.6),
                    duration=30.0,
                    description="Recovery period"
                )
                
        elif scenario_name == "recovery_patterns":
            phase = (time.time() - self.start_time) // 100  # Three phases
            if phase == 0:  # Baseline
                return EnvironmentalStimulus(
                    type=EnvironmentType.PEACEFUL.value,
                    intensity=0.5,
                    duration=30.0,
                    description="Baseline period"
                )
            elif phase == 1:  # Stress
                return EnvironmentalStimulus(
                    type=EnvironmentType.CHAOTIC.value,
                    intensity=0.8,
                    duration=30.0,
                    description="Stress period"
                )
            else:  # Recovery
                return EnvironmentalStimulus(
                    type=EnvironmentType.NURTURING.value,
                    intensity=0.6,
                    duration=30.0,
                    description="Recovery period"
                )
        
        # Default to random stimulus
        return EnvironmentalStimulus(
            type=random.choice(list(EnvironmentType)).value,
            intensity=random.uniform(0.3, 0.8),
            duration=30.0,
            description="Random environmental change"
        )
        
    def _get_mother_response(self, env_state: dict) -> dict:
        """Get mother's response to current environment state"""
        # Create emotional vector from environment
        emotional_vector = torch.tensor(env_state['emotional_atmosphere'])
        
        # Generate context description
        context = self._generate_context_description(env_state)
        
        # Get mother's response
        response = self.mother.generate_stimulus(
            stage=self.mother.stage_prompts.keys().__iter__().__next__(),  # Use first stage
            child_response=context
        )
        
        return {
            'text': response['text'],
            'emotional_vector': response['emotional_vector'].tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
    def _generate_context_description(self, env_state: dict) -> str:
        """Generate a description of the environment for the mother"""
        descriptions = []
        
        # Light level description
        if env_state['light_level'] < 0.3:
            descriptions.append("The room is very dark")
        elif env_state['light_level'] < 0.6:
            descriptions.append("The lighting is dim")
        else:
            descriptions.append("The room is well lit")
            
        # Noise level description
        if env_state['noise_level'] > 0.7:
            descriptions.append("There are loud noises")
        elif env_state['noise_level'] > 0.4:
            descriptions.append("There are moderate sounds")
        else:
            descriptions.append("It is quiet")
            
        # Temperature description
        if env_state['temperature'] < 0.3:
            descriptions.append("The environment is cold")
        elif env_state['temperature'] > 0.7:
            descriptions.append("The environment is warm")
            
        # Chaos level description
        if env_state['chaos_level'] > 0.7:
            descriptions.append("The environment is chaotic")
        elif env_state['chaos_level'] < 0.3:
            descriptions.append("The environment is peaceful")
            
        return " and ".join(descriptions) + "."
        
    def _record_interaction(self, env_state, mother_response):
        """Record and display interaction details"""
        # Handle both dict and EnvironmentState object types
        if hasattr(env_state, 'to_dict'):
            env_state_dict = env_state.to_dict()
            light_level = env_state.light_level
            noise_level = env_state.noise_level
            temperature = env_state.temperature
            chaos_level = env_state.chaos_level
            active_stimuli = env_state.active_stimuli
        else:
            env_state_dict = env_state
            light_level = env_state.get('light_level', 0.5)
            noise_level = env_state.get('noise_level', 0.3)
            temperature = env_state.get('temperature', 0.5)
            chaos_level = env_state.get('chaos_level', 0.2)
            active_stimuli = env_state.get('active_stimuli', [])

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "environment_state": env_state_dict,
            "mother_response": mother_response
        }
        self.experiment_data.append(interaction)
        
        # Display formatted interaction
        print("\n" + "="*50)
        print(f"Environment State:")
        print(f"  Light: {light_level:.2f}")
        print(f"  Noise: {noise_level:.2f}")
        print(f"  Temperature: {temperature:.2f}")
        print(f"  Chaos: {chaos_level:.2f}")
        
        print("\nActive Stimuli:")
        if isinstance(active_stimuli, list):
            for stimulus in active_stimuli[-3:]:  # Show last 3 stimuli
                if isinstance(stimulus, dict):
                    print(f"  - {stimulus.get('description', 'Unknown')} (Type: {stimulus.get('type', 'Unknown')}, Intensity: {stimulus.get('intensity', 0.0):.2f})")
                else:
                    print(f"  - {stimulus.description} (Type: {stimulus.type}, Intensity: {stimulus.intensity:.2f})")
        
        print("\nMother's Response:")
        if isinstance(mother_response, dict):
            if 'text' in mother_response and mother_response['text']:
                print(f"  Message: {mother_response['text']}")
            if 'content' in mother_response and mother_response['content']:
                print(f"  Message: {mother_response['content']}")
            if 'emotional_vector' in mother_response:
                emotions = mother_response['emotional_vector']
                print(f"  Emotional State:")
                print(f"    Joy: {emotions[0]:.2f}")
                print(f"    Trust: {emotions[1]:.2f}")
                print(f"    Fear: {emotions[2]:.2f}")
                print(f"    Surprise: {emotions[3]:.2f}")
        
        print("="*50 + "\n")
        
    def _save_experiment_results(self, experiment_dir: str):
        """Save all experiment data"""
        # Save environment history
        self.environment.save_experiment_data(f"{experiment_dir}/environment_history.json")
        
        # Save mother responses
        with open(f"{experiment_dir}/experiment_data.json", 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
            
        # Generate and save summary
        summary = self._generate_experiment_summary()
        with open(f"{experiment_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
    def _generate_experiment_summary(self) -> dict:
        """Generate a summary of the experiment results"""
        return {
            'total_interactions': len(self.experiment_data),
            'start_time': self.experiment_data[0]['timestamp'] if self.experiment_data else None,
            'end_time': self.experiment_data[-1]['timestamp'] if self.experiment_data else None,
            'emotional_progression': self._analyze_emotional_progression(),
            'response_patterns': self._analyze_response_patterns()
        }
        
    def _analyze_emotional_progression(self) -> dict:
        """Analyze how emotions changed throughout the experiment"""
        if not self.experiment_data:
            return {}
            
        emotional_states = [
            torch.tensor(data['environment_state']['emotional_atmosphere'])
            for data in self.experiment_data
        ]
        emotional_states = torch.stack(emotional_states)
        
        return {
            'mean_joy': emotional_states[:, 0].mean().item(),
            'mean_trust': emotional_states[:, 1].mean().item(),
            'mean_fear': emotional_states[:, 2].mean().item(),
            'mean_surprise': emotional_states[:, 3].mean().item(),
            'variance_joy': emotional_states[:, 0].var().item(),
            'variance_trust': emotional_states[:, 1].var().item(),
            'variance_fear': emotional_states[:, 2].var().item(),
            'variance_surprise': emotional_states[:, 3].var().item()
        }
        
    def _analyze_response_patterns(self) -> dict:
        """Analyze patterns in mother's responses"""
        if not self.experiment_data:
            return {}
            
        responses = [data['mother_response'] for data in self.experiment_data]
        
        # Analyze emotional vectors in responses
        emotional_vectors = torch.tensor([r['emotional_vector'] for r in responses])
        
        return {
            'mean_response_joy': emotional_vectors[:, 0].mean().item(),
            'mean_response_trust': emotional_vectors[:, 1].mean().item(),
            'mean_response_fear': emotional_vectors[:, 2].mean().item(),
            'mean_response_surprise': emotional_vectors[:, 3].mean().item(),
            'response_consistency': emotional_vectors.var(dim=0).mean().item()
        }

def main():
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Define scenarios to run
    scenarios = [
        "peaceful_to_chaotic",
        "light_to_dark",
        "social_isolation",          # Test mother's response to isolation
        "overstimulation",          # Test handling of overwhelming environments
        "threat_response",          # Test protective behaviors
        "emotional_regulation",      # Test emotional stability
        "adaptive_nurturing",       # Test adaptation to changing needs
        "environmental_learning",    # Test learning from environment
        "stress_resilience",        # Test handling of prolonged stress
        "recovery_patterns"         # Test recovery after difficult situations
    ]
    
    # Run each scenario
    for scenario in scenarios:
        print(f"\nStarting scenario: {scenario}")
        runner.run_scenario(scenario, duration=300)  # 5 minutes per scenario
        print(f"Completed scenario: {scenario}")
        
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main() 