# emotional_development_system.py
# Created by Christopher Celaya
# Advanced emotional development and regulation system for neural child

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from datetime import datetime

class EmotionalCapability(Enum):
    BASIC = "basic"              # Simple emotional responses (0-3 months)
    DIFFERENTIATED = "diff"      # Can distinguish between basic emotions (3-6 months)
    SOCIAL = "social"            # Social referencing begins (6-12 months)
    SELF_AWARE = "self_aware"    # Emergence of self-conscious emotions (12-18 months)
    REGULATED = "regulated"      # Beginning emotional regulation (18-24 months)
    COMPLEX = "complex"          # Complex emotional understanding (2+ years)

@dataclass
class EmotionalState:
    """Comprehensive emotional state representation"""
    # Primary emotions (always present)
    joy: float
    trust: float
    fear: float
    surprise: float
    
    # Secondary emotions (develop over time)
    curiosity: Optional[float] = 0.0
    frustration: Optional[float] = 0.0
    attachment: Optional[float] = 0.0
    pride: Optional[float] = 0.0
    
    # Emotional regulation capacity
    regulation_strength: float = 0.1
    recovery_rate: float = 0.1
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.joy, self.trust, self.fear, self.surprise,
            self.curiosity, self.frustration, self.attachment, self.pride
        ])

class EmotionalDevelopmentSystem(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.capability = EmotionalCapability.BASIC
        self.emotional_state = EmotionalState(
            joy=0.3, trust=0.3, fear=0.1, surprise=0.2
        )
        
        # Set default tensor type
        torch.set_default_tensor_type(torch.FloatTensor)
        
        # Neural networks for emotional processing
        self.emotional_processor = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        ).to(device).float()
        
        # Attachment system
        self.attachment_network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(device).float()
        
        # Emotional memory
        self.emotional_memory = []
        self.max_memory_size = 100
        
        # Development tracking
        self.development_history = []
        self.last_assessment = datetime.now()
        
    def process_stimulus(self, 
                        mother_response: Dict[str, float],
                        environmental_state: Dict[str, float]) -> Dict[str, float]:
        """Process incoming stimulus and update emotional state"""
        # Convert inputs to tensors
        mother_emotions = torch.tensor([
            mother_response.get('joy', 0.5),
            mother_response.get('trust', 0.5),
            mother_response.get('fear', 0.1),
            mother_response.get('surprise', 0.2)
        ], device=self.device, dtype=torch.float32)
        
        env_tensor = torch.tensor([
            environmental_state.get('safety', 0.5),
            environmental_state.get('novelty', 0.3),
            environmental_state.get('comfort', 0.5),
            environmental_state.get('stimulation', 0.4)
        ], device=self.device, dtype=torch.float32)
        
        # Calculate attachment influence
        attachment_influence = self.attachment_network(mother_emotions)
        
        # Process emotional response based on capability level
        if self.capability == EmotionalCapability.BASIC:
            # Simple mirroring with high influence from mother
            new_state = self._basic_emotional_response(mother_emotions, env_tensor)
        else:
            # More complex emotional processing
            new_state = self._process_complex_emotions(mother_emotions, env_tensor, attachment_influence)
        
        # Apply emotional regulation
        regulated_state = self._apply_regulation(new_state)
        
        # Update emotional state
        self._update_emotional_state(regulated_state)
        
        # Record for development tracking
        self._record_emotional_event(mother_emotions, env_tensor, regulated_state)
        
        return self._format_emotional_response()
    
    def _basic_emotional_response(self, mother_emotions: torch.Tensor, 
                                env_tensor: torch.Tensor) -> torch.Tensor:
        """Handle basic emotional responses for early development"""
        # High influence from mother's emotions
        mother_influence = 0.7
        env_influence = 0.3
        
        # Simple weighted combination
        basic_response = (mother_emotions * mother_influence + 
                         env_tensor * env_influence)
        
        # Add small random variations for naturality
        noise = torch.randn_like(basic_response) * 0.1
        return torch.clamp(basic_response + noise, 0, 1)
    
    def _process_complex_emotions(self, mother_emotions: torch.Tensor,
                                env_tensor: torch.Tensor,
                                attachment: torch.Tensor) -> torch.Tensor:
        """Process emotions with more sophistication for later stages"""
        # Combine all inputs
        combined_input = torch.cat([
            mother_emotions,
            env_tensor
        ])
        
        # Process through emotional network
        processed = self.emotional_processor(combined_input)
        
        # Apply attachment modulation
        attachment_strength = float(attachment)
        processed = processed * (0.5 + 0.5 * attachment_strength)
        
        # Add developmental noise (reduces with maturity)
        maturity_factor = min(1.0, len(self.development_history) / 100)
        noise_scale = 0.2 * (1 - maturity_factor)
        noise = torch.randn_like(processed) * noise_scale
        
        return torch.clamp(processed + noise, 0, 1)
    
    def _apply_regulation(self, emotional_state: torch.Tensor) -> torch.Tensor:
        """Apply emotional regulation based on development level"""
        # Ensure we're only working with primary emotions for regulation
        primary_emotions = emotional_state[:4] if len(emotional_state) > 4 else emotional_state
        
        if self.capability.value >= EmotionalCapability.REGULATED.value:
            # Apply learned regulation
            regulation_strength = self.emotional_state.regulation_strength
            baseline = torch.tensor([0.5, 0.5, 0.2, 0.3], device=self.device, dtype=torch.float32)
            regulated = (primary_emotions * (1 - regulation_strength) + 
                       baseline * regulation_strength)
        else:
            # Basic regulation through recovery rate
            recovery_rate = self.emotional_state.recovery_rate
            current = torch.tensor([
                self.emotional_state.joy,
                self.emotional_state.trust,
                self.emotional_state.fear,
                self.emotional_state.surprise
            ], device=self.device, dtype=torch.float32)
            regulated = primary_emotions * (1 - recovery_rate) + current * recovery_rate
        
        # If we have secondary emotions, keep them unchanged
        if len(emotional_state) > 4:
            regulated = torch.cat([regulated, emotional_state[4:]])
            
        return regulated
    
    def _update_emotional_state(self, new_state: torch.Tensor):
        """Update the emotional state with new values"""
        self.emotional_state = EmotionalState(
            joy=float(new_state[0]),
            trust=float(new_state[1]),
            fear=float(new_state[2]),
            surprise=float(new_state[3]),
            curiosity=float(new_state[4]) if len(new_state) > 4 else self.emotional_state.curiosity,
            frustration=float(new_state[5]) if len(new_state) > 5 else self.emotional_state.frustration,
            attachment=float(new_state[6]) if len(new_state) > 6 else self.emotional_state.attachment,
            pride=float(new_state[7]) if len(new_state) > 7 else self.emotional_state.pride,
            regulation_strength=self.emotional_state.regulation_strength,
            recovery_rate=self.emotional_state.recovery_rate
        )
    
    def _record_emotional_event(self, mother_emotions: torch.Tensor,
                              env_tensor: torch.Tensor,
                              response: torch.Tensor):
        """Record emotional event for development tracking"""
        event = {
            'timestamp': datetime.now(),
            'mother_emotions': mother_emotions.cpu().tolist(),
            'environment': env_tensor.cpu().tolist(),
            'response': response.cpu().tolist(),
            'capability_level': self.capability.value,
            'regulation_strength': self.emotional_state.regulation_strength,
            'recovery_rate': self.emotional_state.recovery_rate
        }
        
        self.emotional_memory.append(event)
        if len(self.emotional_memory) > self.max_memory_size:
            self.emotional_memory = self.emotional_memory[-self.max_memory_size:]
    
    def _format_emotional_response(self) -> Dict[str, float]:
        """Format the current emotional state for output"""
        return {
            'joy': self.emotional_state.joy,
            'trust': self.emotional_state.trust,
            'fear': self.emotional_state.fear,
            'surprise': self.emotional_state.surprise,
            'curiosity': self.emotional_state.curiosity,
            'frustration': self.emotional_state.frustration,
            'attachment': self.emotional_state.attachment,
            'pride': self.emotional_state.pride,
            'regulation_level': self.emotional_state.regulation_strength,
            'emotional_capability': self.capability.value
        }
    
    def assess_development(self) -> Dict[str, float]:
        """Assess emotional development progress"""
        if len(self.emotional_memory) < 10:
            return {
                'stability': 0.1,
                'regulation': 0.1,
                'attachment': 0.1,
                'complexity': 0.1
            }
        
        # Calculate emotional stability
        recent_responses = [event['response'] for event in self.emotional_memory[-10:]]
        stability = 1.0 - np.std(recent_responses, axis=0).mean()
        
        # Calculate regulation effectiveness
        regulation_success = np.mean([
            1.0 - abs(response[2] - 0.2)  # How well fear is regulated
            for response in recent_responses
        ])
        
        # Calculate attachment security
        attachment_score = np.mean([
            event['response'][1]  # Trust level
            for event in self.emotional_memory[-10:]
        ])
        
        # Calculate emotional complexity
        complexity = min(1.0, len(set(
            event['response'][3]  # Surprise variations
            for event in self.emotional_memory[-10:]
        )) / 5.0)
        
        assessment = {
            'stability': float(stability),
            'regulation': float(regulation_success),
            'attachment': float(attachment_score),
            'complexity': float(complexity)
        }
        
        self.development_history.append({
            'timestamp': datetime.now(),
            'metrics': assessment
        })
        
        return assessment
    
    def update_capability(self, assessment: Dict[str, float]):
        """Update emotional capability based on development assessment"""
        # Calculate overall development score
        development_score = np.mean(list(assessment.values()))
        
        # Update capability level based on development
        if (development_score > 0.8 and 
            self.capability != EmotionalCapability.COMPLEX):
            self._advance_capability()
        
        # Update regulation parameters
        self.emotional_state.regulation_strength = min(
            0.8, 
            self.emotional_state.regulation_strength + 0.1 * assessment['regulation']
        )
        self.emotional_state.recovery_rate = min(
            0.5,
            self.emotional_state.recovery_rate + 0.05 * assessment['stability']
        )
    
    def _advance_capability(self):
        """Advance to the next emotional capability level"""
        current_level = self.capability.value
        next_levels = {
            'basic': EmotionalCapability.DIFFERENTIATED,
            'diff': EmotionalCapability.SOCIAL,
            'social': EmotionalCapability.SELF_AWARE,
            'self_aware': EmotionalCapability.REGULATED,
            'regulated': EmotionalCapability.COMPLEX
        }
        
        if current_level in next_levels:
            self.capability = next_levels[current_level]
            # Record development milestone
            self.development_history.append({
                'timestamp': datetime.now(),
                'event': f'Advanced to {self.capability.value} emotional capability',
                'previous_level': current_level
            }) 