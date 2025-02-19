# environment_simulator.py
# Created by Christopher Celaya
# Simulates a controlled environment for testing mother LLM responses

import torch
import random
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import json
from datetime import datetime

class EnvironmentType(Enum):
    PEACEFUL = "peaceful"
    CHAOTIC = "chaotic"
    DARK = "dark"
    BRIGHT = "bright"
    NOISY = "noisy"
    QUIET = "quiet"
    COLD = "cold"
    WARM = "warm"
    CROWDED = "crowded"
    ISOLATED = "isolated"
    THREATENING = "threatening"
    NURTURING = "nurturing"
    STIMULATING = "stimulating"
    BORING = "boring"
    UNPREDICTABLE = "unpredictable"
    STRUCTURED = "structured"

@dataclass
class EnvironmentalStimulus:
    type: str
    intensity: float  # 0.0 to 1.0
    duration: float  # in seconds
    description: str

class EnvironmentState:
    def __init__(self):
        self.light_level: float = 0.5  # 0.0 (dark) to 1.0 (bright)
        self.noise_level: float = 0.3  # 0.0 (silent) to 1.0 (very loud)
        self.temperature: float = 0.5  # 0.0 (cold) to 1.0 (hot)
        self.chaos_level: float = 0.2  # 0.0 (peaceful) to 1.0 (chaotic)
        self.active_stimuli: List[EnvironmentalStimulus] = []
        self.emotional_atmosphere: torch.Tensor = torch.tensor([0.5, 0.5, 0.2, 0.3])  # joy, trust, fear, surprise

    def to_dict(self) -> Dict:
        return {
            "light_level": self.light_level,
            "noise_level": self.noise_level,
            "temperature": self.temperature,
            "chaos_level": self.chaos_level,
            "emotional_atmosphere": self.emotional_atmosphere.tolist(),
            "active_stimuli": [
                {
                    "type": stim.type,
                    "intensity": stim.intensity,
                    "duration": stim.duration,
                    "description": stim.description
                }
                for stim in self.active_stimuli
            ]
        }

class BlackBoxEnvironment:
    def __init__(self, logger):
        self.logger = logger
        self.state = EnvironmentState()
        self.history: List[Dict] = []
        self.start_time = datetime.now()
        
    def add_stimulus(self, stimulus: EnvironmentalStimulus):
        """Add a new stimulus to the environment"""
        self.state.active_stimuli.append(stimulus)
        self._update_environment_state(stimulus)
        self._log_stimulus(stimulus)
        
    def _update_environment_state(self, stimulus: EnvironmentalStimulus):
        """Update environment state based on new stimulus"""
        # Update basic environmental parameters
        if stimulus.type == EnvironmentType.BRIGHT.value:
            self.state.light_level = min(1.0, self.state.light_level + stimulus.intensity * 0.3)
        elif stimulus.type == EnvironmentType.DARK.value:
            self.state.light_level = max(0.0, self.state.light_level - stimulus.intensity * 0.3)
        elif stimulus.type == EnvironmentType.NOISY.value:
            self.state.noise_level = min(1.0, self.state.noise_level + stimulus.intensity * 0.3)
        elif stimulus.type == EnvironmentType.QUIET.value:
            self.state.noise_level = max(0.0, self.state.noise_level - stimulus.intensity * 0.3)
        elif stimulus.type == EnvironmentType.WARM.value:
            self.state.temperature = min(1.0, self.state.temperature + stimulus.intensity * 0.3)
        elif stimulus.type == EnvironmentType.COLD.value:
            self.state.temperature = max(0.0, self.state.temperature - stimulus.intensity * 0.3)
        elif stimulus.type == EnvironmentType.CHAOTIC.value:
            self.state.chaos_level = min(1.0, self.state.chaos_level + stimulus.intensity * 0.3)
        elif stimulus.type == EnvironmentType.PEACEFUL.value:
            self.state.chaos_level = max(0.0, self.state.chaos_level - stimulus.intensity * 0.3)
            
        # Update emotional atmosphere
        self._update_emotional_atmosphere(stimulus)
        
    def _update_emotional_atmosphere(self, stimulus: EnvironmentalStimulus):
        """Update the emotional atmosphere based on stimulus"""
        # Create emotional impact vector based on stimulus type and intensity
        impact = torch.zeros(4)  # joy, trust, fear, surprise
        
        if stimulus.type == EnvironmentType.PEACEFUL.value:
            impact[0] += 0.3  # joy
            impact[1] += 0.4  # trust
            impact[2] -= 0.2  # fear
        elif stimulus.type == EnvironmentType.CHAOTIC.value:
            impact[2] += 0.4  # fear
            impact[3] += 0.3  # surprise
            impact[1] -= 0.2  # trust
        elif stimulus.type == EnvironmentType.DARK.value:
            impact[2] += 0.3  # fear
            impact[1] -= 0.2  # trust
        elif stimulus.type == EnvironmentType.BRIGHT.value:
            impact[0] += 0.2  # joy
            impact[3] += 0.1  # surprise
        
        # Apply impact with intensity scaling
        impact *= stimulus.intensity
        
        # Update emotional atmosphere with decay
        decay = 0.9
        self.state.emotional_atmosphere = torch.clamp(
            decay * self.state.emotional_atmosphere + (1 - decay) * impact,
            0.0, 1.0
        )
        
    def _log_stimulus(self, stimulus: EnvironmentalStimulus):
        """Log stimulus and its effects"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "stimulus": {
                "type": stimulus.type,
                "intensity": stimulus.intensity,
                "duration": stimulus.duration,
                "description": stimulus.description
            },
            "resulting_state": self.state.to_dict()
        }
        self.history.append(event)
        self.logger.log_development(f"Environment Update: {json.dumps(event, indent=2)}")
        
    def get_current_state(self) -> Dict:
        """Get the current state of the environment"""
        return self.state.to_dict()
        
    def save_experiment_data(self, path: str):
        """Save the experiment data to a file"""
        experiment_data = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "history": self.history
        }
        
        with open(path, 'w') as f:
            json.dump(experiment_data, f, indent=2) 