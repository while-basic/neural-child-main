# emotional_regulation.py
# Description: Emotional regulation system for neural child development
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from logger import DevelopmentLogger

class EmotionalState:
    def __init__(self, device='cuda'):
        self.device = device
        # Use only 4 primary emotions: joy, trust, fear, surprise.
        self.primary_emotions = nn.ParameterDict({
            'joy': nn.Parameter(torch.tensor(0.0, device=device)),
            'trust': nn.Parameter(torch.tensor(0.0, device=device)),
            'fear': nn.Parameter(torch.tensor(0.0, device=device)),
            'surprise': nn.Parameter(torch.tensor(0.0, device=device))
        })
        
        self.complex_emotions = {
            'love': {'joy': 0.6, 'trust': 0.4},
            'guilt': {'fear': 0.5, 'surprise': 0.5},
            'pride': {'joy': 0.7, 'fear': 0.3},
            'shame': {'trust': 0.6, 'surprise': 0.4},
            'anxiety': {'fear': 0.7, 'surprise': 0.3},
            'contentment': {'joy': 0.5, 'trust': 0.5},
            'rejection': {'fear': 0.4, 'surprise': 0.6},
            'excitement': {'joy': 0.5, 'surprise': 0.5}
        }
        
        self.stability_window = deque(maxlen=100)
        self.baseline = {k: 0.5 for k in self.primary_emotions.keys()}
        
    def update(self, emotional_input: dict, learning_rate: float = 0.1) -> None:
        for emotion, value in emotional_input.items():
            if emotion in self.primary_emotions:
                current = self.primary_emotions[emotion].item()
                delta = (value - current) * learning_rate
                noise = torch.randn(1, device=self.device).item() * 0.05
                new_value = torch.clamp(current + delta + noise, 0.0, 1.0)
                self.primary_emotions[emotion].data = torch.tensor(new_value, device=self.device)
                
        total_change = sum(abs(self.primary_emotions[k].item() - self.baseline[k]) for k in self.primary_emotions.keys())
        self.stability_window.append(total_change)
        
    def get_complex_emotion(self, emotion_name: str) -> float:
        if emotion_name not in self.complex_emotions:
            return 0.0
        composition = self.complex_emotions[emotion_name]
        intensity = sum(
            self.primary_emotions[primary].item() * weight 
            for primary, weight in composition.items()
        )
        return float(torch.clamp(torch.tensor(intensity), 0.0, 1.0))
    
    def get_dominant_emotion(self):
        primary_intensities = {name: self.primary_emotions[name].item() for name in self.primary_emotions.keys()}
        complex_intensities = {name: self.get_complex_emotion(name) for name in self.complex_emotions.keys()}
        all_emotions = {**primary_intensities, **complex_intensities}
        dominant = max(all_emotions.items(), key=lambda x: x[1])
        return dominant
    
    def get_emotional_stability(self) -> float:
        if not self.stability_window:
            return 1.0
        recent_volatility = sum(self.stability_window) / len(self.stability_window)
        stability = 1.0 - min(recent_volatility, 1.0)
        return float(stability)
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.primary_emotions[emotion].item() for emotion in sorted(self.primary_emotions.keys())], device=self.device)
    
    def from_tensor(self, tensor: torch.Tensor) -> None:
        sorted_emotions = sorted(self.primary_emotions.keys())
        for i, emotion in enumerate(sorted_emotions):
            self.primary_emotions[emotion].data = tensor[i]

class EmotionalRegulation(nn.Module):
    def __init__(self, emotion_dim=4, context_window=5, memory_dim=32, device='cpu'):
        super().__init__()
        self.emotion_dim = emotion_dim
        self.context_window = context_window
        self.memory_dim = memory_dim
        self.device = device
        
        # Define the missing parameters
        self.trauma_threshold = 1.0
        self.resilience = 1.0
        
        self.context_processor = nn.LSTM(
            input_size=emotion_dim,
            hidden_size=emotion_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(device)
        
        self.stability_net = nn.Sequential(
            nn.Linear(emotion_dim * 2 + memory_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, emotion_dim)
        ).to(device)
        
        self.memory_gate = nn.Sequential(
            nn.Linear(memory_dim + emotion_dim, 64),
            nn.GELU(),
            nn.Linear(64, emotion_dim),
            nn.Sigmoid()
        ).to(device)
        
        self.emotional_history = deque(maxlen=context_window)
        self.baseline = torch.zeros(emotion_dim, device=device)
        
    def to(self, device):
        """Override to method to ensure all components are moved to the same device"""
        super().to(device)
        self.device = device
        self.context_processor.to(device)
        self.stability_net.to(device)
        self.memory_gate.to(device)
        self.baseline = self.baseline.to(device)
        # Move emotional history to device
        if self.emotional_history:
            self.emotional_history = deque(
                [e.to(device) for e in self.emotional_history],
                maxlen=self.context_window
            )
        return self
        
    def update_baseline(self):
        if self.emotional_history:
            recent_emotions = torch.stack(list(self.emotional_history))
            alpha = 0.1
            self.baseline = alpha * recent_emotions.mean(dim=0) + (1 - alpha) * self.baseline
            
    def detect_trauma(self, emotional_state):
        intensity = torch.norm(emotional_state - self.baseline)
        duration = len([e for e in self.emotional_history if torch.norm(e - self.baseline) > 0.7])
        return {
            'is_traumatic': intensity > self.trauma_threshold,
            'duration': duration,
            'intensity': intensity.item()
        }
        
    def compute_regulation_strength(self, emotional_state):
        deviation = torch.abs(emotional_state - self.baseline)
        return torch.sigmoid(deviation * self.resilience)
        
    def regulate(self, emotional_state, stimulus, memory_context=None):
        emotional_state = emotional_state.to(self.device)
        if memory_context is not None:
            memory_context = memory_context.to(self.device)
            
        # Ensure minimum emotional values
        emotional_state = torch.clamp(emotional_state, min=0.1, max=1.0)
            
        if len(self.emotional_history) >= 2:
            context_tensor = torch.stack(list(self.emotional_history))
            context_output, _ = self.context_processor(context_tensor.unsqueeze(0))
            context_embedding = context_output[0, -1]
        else:
            context_embedding = torch.zeros(self.emotion_dim * 2, device=self.device)
            
        if memory_context is not None:
            memory_influence = self.memory_gate(torch.cat([memory_context, emotional_state], dim=-1))
            context_embedding = context_embedding * memory_influence
        else:
            memory_influence = None
            
        combined_input = torch.cat([
            context_embedding,
            memory_context if memory_context is not None else torch.zeros(self.memory_dim, device=self.device)
        ], dim=-1)
        
        regulated_response = self.stability_net(combined_input)
        
        # Apply minimum thresholds and smooth regulation
        base_emotions = torch.tensor([0.2, 0.2, 0.1, 0.1], device=self.device)  # Base levels for joy, trust, fear, surprise
        regulated_response = torch.max(regulated_response, base_emotions)
        
        # Smooth the transition
        alpha = 0.3  # Smoothing factor
        new_state = alpha * regulated_response + (1 - alpha) * emotional_state
        new_state = torch.clamp(new_state, 0.1, 1.0)  # Ensure minimum values
        
        self.emotional_history.append(emotional_state.detach())
        
        return {
            'emotional_state': new_state,
            'context_influence': context_embedding,
            'memory_influence': memory_influence
        }

class EmotionalRegulation:
    def __init__(self, logger: DevelopmentLogger):
        """Initialize the emotional regulation system."""
        self.logger = logger
        self.emotional_state = torch.zeros(4)  # [joy, trust, fear, surprise]
        self.baseline_state = torch.tensor([0.5, 0.5, 0.2, 0.3])  # Default balanced state
        self.regulation_history = []
        self.emotional_memory = []
        
    def update_emotional_state(self, stimulus: torch.Tensor, intensity: float = 1.0) -> Dict:
        """Update emotional state based on new stimulus."""
        try:
            # Apply stimulus with intensity scaling
            impact = stimulus * intensity
            
            # Calculate emotional response with decay
            decay = 0.9  # Emotional decay factor
            self.emotional_state = (
                self.emotional_state * decay + impact * (1 - decay)
            )
            
            # Normalize emotional state
            self.emotional_state = torch.clamp(
                self.emotional_state,
                min=0.0,
                max=1.0
            )
            
            # Log the update
            self.logger.log_emotional_update({
                'stimulus': stimulus.cpu().tolist(),
                'intensity': intensity,
                'resulting_state': self.emotional_state.cpu().tolist()
            })
            
            # Store in history
            self.regulation_history.append({
                'timestamp': datetime.now().isoformat(),
                'stimulus': stimulus.cpu().tolist(),
                'intensity': intensity,
                'state': self.emotional_state.cpu().tolist()
            })
            
            return {
                'success': True,
                'emotional_state': self.emotional_state.cpu().tolist(),
                'intensity': intensity
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'update_emotional_state',
                'stimulus': stimulus.cpu().tolist() if stimulus is not None else None,
                'intensity': intensity
            })
            return {
                'success': False,
                'error': str(e)
            }
    
    def regulate_emotion(self, target_state: Optional[torch.Tensor] = None) -> Dict:
        """Attempt to regulate current emotional state towards target or baseline."""
        try:
            if target_state is None:
                target_state = self.baseline_state
                
            # Calculate regulation vector
            regulation = target_state - self.emotional_state
            
            # Apply gradual regulation
            regulation_rate = 0.2
            self.emotional_state = self.emotional_state + regulation * regulation_rate
            
            # Normalize emotional state
            self.emotional_state = torch.clamp(
                self.emotional_state,
                min=0.0,
                max=1.0
            )
            
            # Log regulation attempt
            self.logger.log_emotional_regulation({
                'target_state': target_state.cpu().tolist(),
                'regulation_applied': regulation.cpu().tolist(),
                'resulting_state': self.emotional_state.cpu().tolist()
            })
            
            return {
                'success': True,
                'emotional_state': self.emotional_state.cpu().tolist(),
                'regulation_applied': regulation.cpu().tolist()
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'regulate_emotion',
                'target_state': target_state.cpu().tolist() if target_state is not None else None
            })
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_emotional_event(self, event: Dict[str, Any]) -> Dict:
        """Process an emotional event and update state accordingly."""
        try:
            # Extract event details
            intensity = event.get('intensity', 1.0)
            valence = event.get('valence', 0.0)  # -1 to 1
            arousal = event.get('arousal', 0.0)  # 0 to 1
            
            # Convert valence and arousal to emotional components
            stimulus = torch.tensor([
                max(0, valence),  # joy
                max(0, valence) * 0.8,  # trust
                max(0, -valence) * arousal,  # fear
                arousal * 0.5  # surprise
            ])
            
            # Update emotional state
            result = self.update_emotional_state(stimulus, intensity)
            
            # Store event in emotional memory
            self.emotional_memory.append({
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'response': result
            })
            
            return result
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'process_emotional_event',
                'event': event
            })
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_emotional_state(self) -> torch.Tensor:
        """Get current emotional state."""
        return self.emotional_state.clone()
    
    def get_regulation_history(self) -> List[Dict]:
        """Get history of emotional regulation attempts."""
        return self.regulation_history
    
    def get_emotional_stability(self) -> float:
        """Calculate emotional stability based on recent history."""
        try:
            if len(self.regulation_history) < 2:
                return 1.0
                
            # Get recent states
            recent_states = [
                torch.tensor(entry['state'])
                for entry in self.regulation_history[-10:]
            ]
            
            # Calculate stability as inverse of state variance
            state_tensor = torch.stack(recent_states)
            variance = torch.var(state_tensor, dim=0).mean().item()
            stability = 1.0 / (1.0 + variance)
            
            return stability
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'get_emotional_stability'
            })
            return 0.0
    
    def analyze_emotional_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in emotional regulation history."""
        try:
            if not self.regulation_history:
                return {
                    'average_state': None,
                    'stability': 1.0,
                    'dominant_emotion': None,
                    'regulation_effectiveness': None
                }
            
            # Calculate average emotional state
            states = [
                torch.tensor(entry['state'])
                for entry in self.regulation_history
            ]
            average_state = torch.stack(states).mean(dim=0)
            
            # Calculate stability
            stability = self.get_emotional_stability()
            
            # Identify dominant emotion
            emotion_labels = ['joy', 'trust', 'fear', 'surprise']
            dominant_idx = torch.argmax(average_state).item()
            dominant_emotion = emotion_labels[dominant_idx]
            
            # Calculate regulation effectiveness
            if len(states) >= 2:
                state_changes = torch.stack([
                    states[i] - states[i-1]
                    for i in range(1, len(states))
                ])
                effectiveness = torch.mean(torch.abs(state_changes)).item()
            else:
                effectiveness = None
            
            return {
                'average_state': average_state.cpu().tolist(),
                'stability': stability,
                'dominant_emotion': dominant_emotion,
                'regulation_effectiveness': effectiveness
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'analyze_emotional_patterns'
            })
            return {
                'error': str(e)
            }
    
    def reset_emotional_state(self):
        """Reset emotional state to baseline."""
        self.emotional_state = self.baseline_state.clone()
        
    def set_baseline_state(self, new_baseline: torch.Tensor):
        """Set a new baseline emotional state."""
        self.baseline_state = torch.clamp(new_baseline, min=0.0, max=1.0)
        
    def get_emotional_memory(self) -> List[Dict]:
        """Get emotional memory history."""
        return self.emotional_memory
        
    def clear_history(self):
        """Clear regulation history and emotional memory."""
        self.regulation_history = []
        self.emotional_memory = []
