# digital_child.py
# Description: Main digital child class integrating all components
# Created by: Christopher Celaya

import torch
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from heartbeat_system import HeartbeatSystem
from emotional_memory_system import EmotionalMemorySystem
from integrated_brain import IntegratedBrain, DevelopmentalStage

class DigitalChild:
    def __init__(self, model_name: str = "llama3"):
        """Initialize the digital child with all components.
        
        Args:
            model_name (str): Name of the language model to use
        """
        # Initialize brain
        self.brain = IntegratedBrain()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize state
        self.emotional_state = torch.zeros(4, device=self.device)  # [joy, trust, fear, surprise]
        self.attention = 0.5
        self.arousal = 0.3
        self.developmental_stage = DevelopmentalStage.NEWBORN
        
        # Initialize memory systems
        self.emotional_memory = EmotionalMemorySystem()
        
        # Initialize heartbeat system
        self.heartbeat = HeartbeatSystem()
        
        # Track interaction history
        self.interaction_history = []
        
    def process_interaction(self, message: str, tone: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Process an interaction with the child.
        
        Args:
            message (str): The interaction message
            tone (Dict[str, float], optional): Emotional tone of the message
            
        Returns:
            Dict containing response information
        """
        # Update emotional state based on message
        if tone:
            self.emotional_state = torch.tensor([
                tone.get('joy', 0.0),
                tone.get('trust', 0.0),
                tone.get('fear', 0.0),
                tone.get('surprise', 0.0)
            ], device=self.device)
        
        # Process through brain
        brain_response = self.brain.process_emotions(
            torch.tensor([0.0]),  # Placeholder for features
            self.emotional_state
        )
        
        # Update heartbeat
        heartbeat_response = self.heartbeat.update_from_emotional_state({
            'joy': float(self.emotional_state[0]),
            'trust': float(self.emotional_state[1]),
            'fear': float(self.emotional_state[2]),
            'surprise': float(self.emotional_state[3]),
            'attention': self.attention,
            'arousal': self.arousal
        })
        
        # Record interaction
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'emotional_state': self.emotional_state.cpu().tolist(),
            'brain_response': brain_response,
            'heartbeat': heartbeat_response
        }
        self.interaction_history.append(interaction)
        
        return {
            'brain_state': self.brain.get_brain_state(),
            'heartbeat': heartbeat_response,
            'emotional_state': self.emotional_state.cpu().tolist(),
            'developmental_stage': self.developmental_stage.name
        }
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the digital child.
        
        Returns:
            Dict containing current state information
        """
        return {
            'brain_state': self.brain.get_brain_state(),
            'heartbeat': self.heartbeat.get_current_heartbeat(),
            'emotional_state': self.emotional_state.cpu().tolist(),
            'developmental_stage': self.developmental_stage.name,
            'attention': self.attention,
            'arousal': self.arousal
        }
        
    def save_state(self, path: str):
        """Save current state to file.
        
        Args:
            path (str): Path to save state
        """
        state = {
            'emotional_state': self.emotional_state.cpu().tolist(),
            'attention': self.attention,
            'arousal': self.arousal,
            'developmental_stage': self.developmental_stage.name,
            'interaction_history': self.interaction_history
        }
        
        # Save brain state
        self.brain.save_brain_state(path + '_brain')
        
        # Save other state components
        torch.save(state, path)
        
    def load_state(self, path: str):
        """Load state from file.
        
        Args:
            path (str): Path to load state from
        """
        # Load brain state
        self.brain.load_brain_state(path + '_brain')
        
        # Load other state components
        state = torch.load(path)
        self.emotional_state = torch.tensor(state['emotional_state'], device=self.device)
        self.attention = state['attention']
        self.arousal = state['arousal']
        self.developmental_stage = DevelopmentalStage[state['developmental_stage']]
        self.interaction_history = state['interaction_history'] 