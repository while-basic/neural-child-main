# brain_state.py
# Description: Brain state management for the neural child development system
# Created by: Christopher Celaya

import torch
from typing import Dict, Any

class BrainState:
    def __init__(self):
        """Initialize brain state with default values"""
        self.emotional_state = {
            'joy': 0.5,
            'trust': 0.5,
            'fear': 0.2,
            'surprise': 0.3,
            'sadness': 0.2,
            'anger': 0.1,
            'disgust': 0.1,
            'anticipation': 0.4,
            'love': 0.3,
            'guilt': 0.1,
            'hope': 0.4,
            'regret': 0.1
        }
        
        # Cognitive state
        self.cognitive_state = {
            'attention': 0.5,
            'memory': 0.3,
            'learning': 0.4,
            'reasoning': 0.2
        }
        
        # Physical state
        self.physical_state = {
            'energy': 1.0,
            'fatigue': 0.0,
            'comfort': 0.8
        }
        
        # Development metrics
        self.development_metrics = {
            'emotional_valence': 0.5,  # Overall emotional state (-1 to 1)
            'arousal': 0.5,  # Level of activation/arousal (0 to 1)
            'cognitive_load': 0.3,  # Current cognitive processing load (0 to 1)
            'social_engagement': 0.4  # Level of social interaction (0 to 1)
        }
        
    def update_emotional_state(self, new_state: Dict[str, float]):
        """Update emotional state values"""
        for emotion, value in new_state.items():
            if emotion in self.emotional_state:
                self.emotional_state[emotion] = max(0.0, min(1.0, value))
                
    def update_cognitive_state(self, new_state: Dict[str, float]):
        """Update cognitive state values"""
        for metric, value in new_state.items():
            if metric in self.cognitive_state:
                self.cognitive_state[metric] = max(0.0, min(1.0, value))
                
    def update_physical_state(self, new_state: Dict[str, float]):
        """Update physical state values"""
        for metric, value in new_state.items():
            if metric in self.physical_state:
                self.physical_state[metric] = max(0.0, min(1.0, value))
                
    def update_development_metrics(self, new_metrics: Dict[str, float]):
        """Update development metrics"""
        for metric, value in new_metrics.items():
            if metric in self.development_metrics:
                self.development_metrics[metric] = max(-1.0 if metric == 'emotional_valence' else 0.0, min(1.0, value))
                
    def get_state(self) -> Dict[str, Any]:
        """Get complete brain state"""
        return {
            'emotional_state': self.emotional_state,
            'cognitive_state': self.cognitive_state,
            'physical_state': self.physical_state,
            'development_metrics': self.development_metrics
        } 