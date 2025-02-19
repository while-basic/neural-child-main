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
import math

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
        
        # Neural networks for emotional processing - ensure 4-dimensional output for basic emotions
        self.emotional_processor = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # Changed to output 4 dimensions for basic emotions
        ).to(device).float()
        
        # Secondary emotion processor for advanced stages
        self.secondary_processor = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # Additional 4 dimensions for secondary emotions
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
        self.environmental_memory = []  # Add environmental memory tracking
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
        # Combine all inputs with enhanced weighting
        combined_input = torch.cat([
            mother_emotions * 1.2,  # Increase mother's influence
            env_tensor * 0.8   # Moderate environmental influence
        ])
        
        # Process primary emotions with enhanced baseline
        primary_emotions = self.emotional_processor(combined_input)
        primary_emotions = torch.clamp(primary_emotions, min=0.2)  # Ensure minimum activity
        
        # Calculate emotional maturity based on development history
        maturity_factor = min(1.0, len(self.development_history) / 100)
        
        # Get capability level for progressive enhancement
        capability_level = 0
        for i, cap in enumerate([
            EmotionalCapability.BASIC,
            EmotionalCapability.DIFFERENTIATED,
            EmotionalCapability.SOCIAL,
            EmotionalCapability.SELF_AWARE,
            EmotionalCapability.REGULATED,
            EmotionalCapability.COMPLEX
        ]):
            if cap == self.capability:
                capability_level = i
                break
        
        # Initialize secondary emotions with progressive development
        if capability_level >= 2:  # SOCIAL and above
            # Calculate secondary emotions with increasing sophistication
            curiosity = torch.sigmoid(primary_emotions[3]) * (1 - primary_emotions[2]) * (0.3 + 0.2 * capability_level)
            frustration = torch.clamp(primary_emotions[2] * (1 - primary_emotions[1]), 0, 1) * (0.2 + 0.2 * capability_level)
            attachment_value = torch.sigmoid(primary_emotions[1] * float(attachment)) * (0.4 + 0.2 * capability_level)
            pride = torch.clamp(primary_emotions[0] * float(attachment) * 0.5, 0, 1) * (0.2 + 0.2 * capability_level)
            
            secondary_emotions = torch.tensor(
                [curiosity, frustration, attachment_value, pride],
                device=self.device
            )
            processed = torch.cat([primary_emotions, secondary_emotions])
        else:
            processed = primary_emotions
        
        # Apply attachment modulation with progressive influence
        attachment_strength = float(attachment) * (0.5 + 0.1 * capability_level)
        
        # Enhanced emotional processing based on capability level
        if self.capability == EmotionalCapability.DIFFERENTIATED:
            # More balanced emotional influence
            processed = processed * (0.6 + 0.4 * attachment_strength)
            # Add emotional persistence with adaptive memory
            if len(self.emotional_memory) > 0:
                last_emotion = torch.tensor(self.emotional_memory[-1]['response'][:len(processed)], device=self.device)
                decay_factor = 0.7 - 0.1 * maturity_factor  # Faster emotional transitions with maturity
                processed = decay_factor * processed + (1 - decay_factor) * last_emotion
                
        elif self.capability == EmotionalCapability.SOCIAL:
            # Enhanced social-emotional processing
            processed = processed * (0.5 + 0.5 * attachment_strength)
            # Enhance social emotions with dynamic modulation
            processed[1] *= 1.2 + 0.1 * maturity_factor  # Enhanced trust development
            processed[3] *= 0.9 + 0.1 * maturity_factor  # More dynamic surprise
            # Adaptive fear response with social buffering
            processed[2] *= max(0.6, 1 - attachment_strength * (0.8 + 0.2 * maturity_factor))
            
        elif self.capability == EmotionalCapability.SELF_AWARE:
            # Enhanced emotional awareness and regulation
            processed = processed * (0.4 + 0.6 * attachment_strength)
            # Enhance secondary emotion development with maturity
            if len(processed) > 4:
                processed[4:] *= maturity_factor * (0.7 + 0.3 * capability_level)
                # Enhanced emotional coherence
                processed[4] *= (1 + processed[3]) * 0.8  # Strong curiosity-surprise link
                processed[5] *= (1 + processed[2]) * 0.7  # Moderate frustration-fear link
                processed[6] *= (1 + processed[1]) * 0.9  # Strong attachment-trust link
                processed[7] *= (1 + processed[0]) * 0.7  # Moderate pride-joy link
            
        elif self.capability == EmotionalCapability.REGULATED:
            # Sophisticated emotional processing with adaptive regulation
            processed = processed * (0.3 + 0.7 * attachment_strength)
            # Apply adaptive regulation with dynamic baselines
            baseline = torch.tensor(
                [0.4, 0.5, 0.2, 0.3] + [0.3] * (len(processed) - 4),
                device=self.device
            )
            regulation_strength = self.emotional_state.regulation_strength
            # Apply differential regulation with maturity influence
            regulation_weights = torch.tensor(
                [0.7, 0.8, 1.1, 0.9] + [0.8] * (len(processed) - 4),
                device=self.device
            ) * (1 - 0.2 * maturity_factor)  # Less regulation with maturity
            processed = processed * (1 - regulation_strength * regulation_weights) + baseline * regulation_strength
            
        elif self.capability == EmotionalCapability.COMPLEX:
            # Full emotional range with sophisticated processing
            processed = processed * (0.2 + 0.8 * attachment_strength)
            # Enhanced emotional memory influence
            if len(self.emotional_memory) > 5:
                recent_emotions = torch.stack([
                    torch.tensor(em['response'][:len(processed)], device=self.device)
                    for em in self.emotional_memory[-5:]
                ])
                # Apply weighted memory influence
                weights = torch.tensor([0.7**i for i in range(5)], device=self.device)
                weights = weights / weights.sum()
                emotional_context = torch.sum(recent_emotions * weights.unsqueeze(1), dim=0)
                processed = 0.6 * processed + 0.4 * emotional_context
        
        # Add developmental noise with adaptive impact
        noise_scale = 0.1 * (1 - maturity_factor)  # Reduced noise with maturity
        noise_weights = torch.ones_like(processed)
        if len(processed) >= 4:
            noise_weights[2] = 0.7  # Moderate noise in fear
            noise_weights[3] = 0.8  # Moderate noise in surprise
        noise = torch.randn_like(processed) * noise_scale * noise_weights
        
        # Apply final stabilization with minimum values
        processed = torch.clamp(processed + noise, 0.2, 1.0)
        
        # Ensure emotional coherence
        if len(processed) > 4:
            # Maintain emotional balance with sophistication
            if processed[2] > 0.7:  # High fear
                processed[0] *= 0.8  # Moderate joy reduction
                processed[1] *= 0.9  # Slight trust reduction
            if processed[0] > 0.8:  # High joy
                processed[2] *= 0.7  # Moderate fear reduction
                processed[5] *= 0.8  # Moderate frustration reduction
            
        return processed
    
    def _apply_regulation(self, emotional_state: torch.Tensor) -> torch.Tensor:
        """Apply emotional regulation based on development level"""
        # Ensure we're only working with primary emotions for regulation
        primary_emotions = emotional_state[:4] if len(emotional_state) > 4 else emotional_state
        
        # Set adaptive baselines based on capability level
        if self.capability.value >= EmotionalCapability.REGULATED.value:
            # More sophisticated regulation with dynamic baselines
            regulation_strength = self.emotional_state.regulation_strength
            
            # Adaptive baselines based on emotional state
            joy_baseline = max(0.3, min(0.6, primary_emotions[0]))
            trust_baseline = max(0.4, min(0.7, primary_emotions[1]))
            fear_baseline = min(0.3, max(0.1, primary_emotions[2]))
            surprise_baseline = min(0.4, max(0.2, primary_emotions[3]))
            
            baseline = torch.tensor(
                [joy_baseline, trust_baseline, fear_baseline, surprise_baseline],
                device=self.device,
                dtype=torch.float32
            )
            
            # Apply differential regulation
            regulation_weights = torch.tensor(
                [0.7, 0.8, 1.2, 0.9],  # Less suppression of positive emotions
                device=self.device
            )
            
            regulated = primary_emotions * (1 - regulation_strength * regulation_weights) + baseline * regulation_strength
            
        else:
            # Basic regulation through recovery rate with minimum values
            recovery_rate = self.emotional_state.recovery_rate
            current = torch.tensor([
                max(0.2, self.emotional_state.joy),
                max(0.2, self.emotional_state.trust),
                min(0.4, max(0.1, self.emotional_state.fear)),
                max(0.2, self.emotional_state.surprise)
            ], device=self.device, dtype=torch.float32)
            
            # Apply gradual regulation with baseline preservation
            regulated = primary_emotions * (1 - recovery_rate) + current * recovery_rate
            
            # Ensure minimum emotional values
            regulated = torch.clamp(regulated, min=0.2)
        
        # If we have secondary emotions, process them separately
        if len(emotional_state) > 4:
            secondary_emotions = emotional_state[4:]
            
            # Apply lighter regulation to secondary emotions
            if self.capability.value >= EmotionalCapability.SELF_AWARE.value:
                regulation_strength = self.emotional_state.regulation_strength * 0.7
                # Maintain some baseline activity in secondary emotions
                secondary_baseline = torch.full_like(secondary_emotions, 0.2)
                regulated_secondary = (
                    secondary_emotions * (1 - regulation_strength) +
                    secondary_baseline * regulation_strength
                )
            else:
                regulated_secondary = secondary_emotions
            
            # Combine regulated primary and secondary emotions
            regulated = torch.cat([regulated, regulated_secondary])
        
        return torch.clamp(regulated, min=0.1, max=1.0)
    
    def _update_emotional_state(self, new_state: torch.Tensor):
        """Update the emotional state with new values and ensure dynamic emotional activity"""
        # Calculate adaptive baselines based on capability level
        capability_level = 0
        for i, cap in enumerate([
            EmotionalCapability.BASIC,
            EmotionalCapability.DIFFERENTIATED,
            EmotionalCapability.SOCIAL,
            EmotionalCapability.SELF_AWARE,
            EmotionalCapability.REGULATED,
            EmotionalCapability.COMPLEX
        ]):
            if cap == self.capability:
                capability_level = i
                break
                
        # Set minimum emotional values based on capability
        min_values = {
            'joy': 0.2 + 0.05 * capability_level,
            'trust': 0.2 + 0.05 * capability_level,
            'fear': max(0.1, 0.2 - 0.02 * capability_level),
            'surprise': 0.2 + 0.03 * capability_level,
            'curiosity': 0.1 if capability_level >= 2 else 0.0,
            'frustration': 0.1 if capability_level >= 2 else 0.0,
            'attachment': 0.1 if capability_level >= 2 else 0.0,
            'pride': 0.1 if capability_level >= 2 else 0.0
        }
        
        # Apply smoothing with adaptive rates
        smoothing_factor = 0.3 - 0.03 * capability_level  # Less smoothing at higher levels
        
        # Update primary emotions with smoothing and minimum values
        new_joy = max(min_values['joy'], 
                     (1 - smoothing_factor) * new_state[0] + 
                     smoothing_factor * self.emotional_state.joy)
        new_trust = max(min_values['trust'], 
                       (1 - smoothing_factor) * new_state[1] + 
                       smoothing_factor * self.emotional_state.trust)
        new_fear = max(min_values['fear'], 
                      (1 - smoothing_factor) * new_state[2] + 
                      smoothing_factor * self.emotional_state.fear)
        new_surprise = max(min_values['surprise'], 
                          (1 - smoothing_factor) * new_state[3] + 
                          smoothing_factor * self.emotional_state.surprise)
        
        # Update secondary emotions if present in new state
        if len(new_state) > 4:
            new_curiosity = max(min_values['curiosity'], 
                              (1 - smoothing_factor) * new_state[4] + 
                              smoothing_factor * self.emotional_state.curiosity)
            new_frustration = max(min_values['frustration'], 
                                (1 - smoothing_factor) * new_state[5] + 
                                smoothing_factor * self.emotional_state.frustration)
            new_attachment = max(min_values['attachment'], 
                               (1 - smoothing_factor) * new_state[6] + 
                               smoothing_factor * self.emotional_state.attachment)
            new_pride = max(min_values['pride'], 
                          (1 - smoothing_factor) * new_state[7] + 
                          smoothing_factor * self.emotional_state.pride)
        else:
            # Maintain existing secondary emotions with some decay
            decay_rate = 0.95
            new_curiosity = max(min_values['curiosity'], 
                              self.emotional_state.curiosity * decay_rate)
            new_frustration = max(min_values['frustration'], 
                                self.emotional_state.frustration * decay_rate)
            new_attachment = max(min_values['attachment'], 
                               self.emotional_state.attachment * decay_rate)
            new_pride = max(min_values['pride'], 
                          self.emotional_state.pride * decay_rate)
        
        # Create new emotional state with updated values
        self.emotional_state = EmotionalState(
            joy=float(new_joy),
            trust=float(new_trust),
            fear=float(new_fear),
            surprise=float(new_surprise),
            curiosity=float(new_curiosity),
            frustration=float(new_frustration),
            attachment=float(new_attachment),
            pride=float(new_pride),
            regulation_strength=self.emotional_state.regulation_strength,
            recovery_rate=self.emotional_state.recovery_rate
        )
        
        # Ensure emotional coherence
        self._maintain_emotional_coherence()
    
    def _maintain_emotional_coherence(self):
        """Ensure emotional coherence by maintaining emotional balance"""
        # Ensure emotional balance with moderation
        if self.emotional_state.fear > 0.7:  # High fear
            self.emotional_state.joy *= 0.8  # Moderate joy reduction
        if self.emotional_state.joy > 0.8:  # High joy
            self.emotional_state.fear *= 0.7  # Moderate fear reduction
    
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
        
        # Record environmental state
        env_state = {
            'timestamp': datetime.now(),
            'state': env_tensor.cpu().tolist(),
            'response': response.cpu().tolist()[:4]  # Only primary emotions for env tracking
        }
        
        self.emotional_memory.append(event)
        self.environmental_memory.append(env_state)
        
        # Maintain memory size limits
        if len(self.emotional_memory) > self.max_memory_size:
            self.emotional_memory = self.emotional_memory[-self.max_memory_size:]
        if len(self.environmental_memory) > self.max_memory_size:
            self.environmental_memory = self.environmental_memory[-self.max_memory_size:]
    
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
        # Calculate overall development score with weighted components
        weights = {
            'stability': 0.3,
            'regulation': 0.3,
            'attachment': 0.2,
            'complexity': 0.2
        }
        development_score = sum(assessment[key] * weight 
                              for key, weight in weights.items())
        
        # Update capability level based on development
        if development_score > 0.7:  # Lower threshold for advancement
            current_level = self.capability.value
            next_levels = {
                EmotionalCapability.BASIC.value: EmotionalCapability.DIFFERENTIATED,
                EmotionalCapability.DIFFERENTIATED.value: EmotionalCapability.SOCIAL,
                EmotionalCapability.SOCIAL.value: EmotionalCapability.SELF_AWARE,
                EmotionalCapability.SELF_AWARE.value: EmotionalCapability.REGULATED,
                EmotionalCapability.REGULATED.value: EmotionalCapability.COMPLEX
            }
            
            if current_level in next_levels:
                # Check specific requirements for each advancement
                should_advance = False
                
                if current_level == EmotionalCapability.BASIC.value:
                    # Basic to Differentiated: Need good stability
                    should_advance = assessment['stability'] > 0.7
                    
                elif current_level == EmotionalCapability.DIFFERENTIATED.value:
                    # Differentiated to Social: Need good attachment and regulation
                    should_advance = (assessment['attachment'] > 0.5 and 
                                    assessment['regulation'] > 0.7)
                    
                elif current_level == EmotionalCapability.SOCIAL.value:
                    # Social to Self-Aware: Need high regulation and complexity
                    should_advance = (assessment['regulation'] > 0.8 and 
                                    assessment['complexity'] > 0.6)
                    
                elif current_level == EmotionalCapability.SELF_AWARE.value:
                    # Self-Aware to Regulated: Need high stability and regulation
                    should_advance = (assessment['stability'] > 0.85 and 
                                    assessment['regulation'] > 0.85)
                    
                elif current_level == EmotionalCapability.REGULATED.value:
                    # Regulated to Complex: Need high scores across all metrics
                    should_advance = all(v > 0.8 for v in assessment.values())
                
                if should_advance:
                    self.capability = next_levels[current_level]
                    # Record development milestone
                    self.development_history.append({
                        'timestamp': datetime.now(),
                        'event': f'Advanced to {self.capability.value} emotional capability',
                        'previous_level': current_level,
                        'assessment_scores': assessment
                    })
        
        # Get numerical capability level (0 to 5)
        capability_level = 0
        for i, cap in enumerate([
            EmotionalCapability.BASIC,
            EmotionalCapability.DIFFERENTIATED,
            EmotionalCapability.SOCIAL,
            EmotionalCapability.SELF_AWARE,
            EmotionalCapability.REGULATED,
            EmotionalCapability.COMPLEX
        ]):
            if cap == self.capability:
                capability_level = i
                break
        
        # Update regulation parameters with more sophisticated progression
        # Increase regulation strength based on successful regulation
        self.emotional_state.regulation_strength = min(
            0.8,
            self.emotional_state.regulation_strength + 
            0.1 * assessment['regulation'] * 
            (1 + 0.2 * capability_level)  # Bonus for higher capabilities
        )
        
        # Increase recovery rate based on stability
        self.emotional_state.recovery_rate = min(
            0.5,
            self.emotional_state.recovery_rate + 
            0.05 * assessment['stability'] * 
            (1 + 0.1 * capability_level)  # Bonus for higher capabilities
        )
        
        # Update emotional complexity
        if capability_level >= 2:  # SOCIAL and above
            # Gradually activate secondary emotions
            self.emotional_state = EmotionalState(
                joy=self.emotional_state.joy,
                trust=self.emotional_state.trust,
                fear=self.emotional_state.fear,
                surprise=self.emotional_state.surprise,
                curiosity=max(0.1, min(0.8, assessment['complexity'] * 0.8)),
                frustration=max(0.1, min(0.6, (1 - assessment['regulation']) * 0.7)),
                attachment=max(0.1, min(0.9, assessment['attachment'] * 0.9)),
                pride=max(0.1, min(0.7, assessment['stability'] * 0.7)),
                regulation_strength=self.emotional_state.regulation_strength,
                recovery_rate=self.emotional_state.recovery_rate
            )
    
    def _process_environmental_influence(self, env_state: Dict[str, float]) -> torch.Tensor:
        """Process environmental influences with enhanced sophistication"""
        # Convert environment state to tensor with default values
        env_factors = [
            env_state.get('noise_level', 0.2),    # Some baseline noise
            env_state.get('light_level', 0.5),    # Moderate light
            env_state.get('temperature', 0.5),    # Comfortable temperature
            env_state.get('social_presence', 0.3), # Some social presence
            env_state.get('physical_comfort', 0.6) # Generally comfortable
        ]
        env_tensor = torch.tensor(env_factors, dtype=torch.float32, device=self.device)
        
        # Calculate time of day influence (circadian rhythm)
        current_time = datetime.now().hour
        circadian_factor = math.sin(2 * math.pi * (current_time - 6) / 24)  # Peak at noon, trough at midnight
        
        # Process environmental factors based on capability level
        if self.capability == EmotionalCapability.BASIC:
            # Simple direct influence of environment with baseline activity
            influence = torch.zeros(4, device=self.device) + 0.2  # Baseline emotional activity
            influence[0] = 0.3 + 0.4 * env_tensor[4]  # Joy influenced by physical comfort
            influence[2] = 0.2 + 0.3 * (1 - env_tensor[4])  # Fear influenced by discomfort
            influence[1] = 0.3 + 0.2 * env_tensor[3]  # Trust influenced by social presence
            influence[3] = 0.2 + 0.3 * env_tensor[1]  # Surprise influenced by light
            
        elif self.capability == EmotionalCapability.DIFFERENTIATED:
            # More nuanced environmental processing with active responses
            influence = torch.zeros(4, device=self.device) + 0.25  # Higher baseline
            influence[0] = 0.3 + 0.4 * env_tensor[4] + 0.2 * env_tensor[1]  # Joy: comfort + light
            influence[1] = 0.3 + 0.4 * env_tensor[3]  # Trust: social presence
            influence[2] = 0.2 + 0.3 * (1 - env_tensor[4]) + 0.2 * env_tensor[0]  # Fear: discomfort + noise
            influence[3] = 0.2 + 0.4 * abs(env_tensor[2] - 0.5)  # Surprise: temperature deviation
            
        else:
            # Sophisticated environmental processing with rich emotional responses
            influence = torch.zeros(8, device=self.device) + 0.3  # Complex baseline
            
            # Primary emotions with enhanced environmental awareness
            influence[0] = 0.3 + 0.3 * env_tensor[4] + 0.2 * env_tensor[1] + 0.2 * circadian_factor  # Joy
            influence[1] = 0.3 + 0.4 * env_tensor[3] + 0.2 * env_tensor[4]  # Trust
            influence[2] = 0.2 + 0.3 * (1 - env_tensor[4]) + 0.2 * env_tensor[0]  # Fear
            influence[3] = 0.2 + 0.3 * abs(env_tensor[2] - 0.5) + 0.2 * env_tensor[0]  # Surprise
            
            # Secondary emotions with sophisticated processing
            if len(influence) > 4:
                # Enhanced secondary emotion development
                influence[4] = 0.3 + 0.3 * env_tensor[1] + 0.3 * env_tensor[3]  # Curiosity
                influence[5] = 0.2 + 0.3 * env_tensor[0] + 0.2 * (1 - env_tensor[4])  # Frustration
                influence[6] = 0.3 + 0.4 * env_tensor[3] + 0.2 * env_tensor[4]  # Attachment
                influence[7] = 0.2 + 0.3 * env_tensor[4] + 0.2 * circadian_factor  # Pride
            
            # Apply circadian modulation with baseline preservation
            influence = influence * (0.7 + 0.3 * (circadian_factor + 1) / 2)
            
            # Environmental adaptation with stability
            if hasattr(self, 'environmental_memory') and len(self.environmental_memory) > 10:
                recent_env = torch.stack([torch.tensor(em['state'], device=self.device) 
                                        for em in self.environmental_memory[-10:]])
                env_stability = 1 - torch.std(recent_env, dim=0).mean()
                influence = influence * (0.6 + 0.4 * env_stability)
        
        # Ensure healthy baseline emotional activity
        return torch.clamp(influence, min=0.2, max=1.0) 