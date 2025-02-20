# moral_network.py
# Description: Neural network for moral reasoning and ethical decision making
# Created by: Christopher Celaya

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from datetime import datetime

class MoralValue:
    """Representation of a moral value"""
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.activation = 0.0
        self.history = []
        
    def update(self, activation: float):
        """Update value activation"""
        self.activation = activation
        self.history.append({
            'activation': activation,
            'timestamp': datetime.now().isoformat()
        })
        
class MoralNetwork(nn.Module):
    """Neural network for moral reasoning and ethical decision making"""
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Core values network
        self.values_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Moral reasoning network
        self.reasoning_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decision network
        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4 basic moral decisions
        )
        
        # Moral attention
        self.moral_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Initialize moral values
        self.moral_values = {
            'care': MoralValue('Care/Harm', 1.0),
            'fairness': MoralValue('Fairness/Cheating', 1.0),
            'loyalty': MoralValue('Loyalty/Betrayal', 0.8),
            'authority': MoralValue('Authority/Subversion', 0.8),
            'sanctity': MoralValue('Sanctity/Degradation', 0.6)
        }
        
        # Value embeddings
        self.value_embeddings = nn.Parameter(
            torch.randn(len(self.moral_values), hidden_dim // 2)
        )
        
        # Decision history
        self.decision_history = []
        
    def forward(self, 
                input_data: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Process input through moral reasoning network"""
        batch_size = input_data.shape[0]
        device = input_data.device
        
        # Process through values network
        values_output = self.values_network(input_data)
        
        # Calculate value activations
        value_activations = torch.matmul(
            values_output,
            self.value_embeddings.t()
        )
        value_activations = torch.sigmoid(value_activations)
        
        # Update moral values
        for i, (name, value) in enumerate(self.moral_values.items()):
            value.update(float(value_activations[0, i].item()))
        
        # Process through reasoning network
        reasoning_output = self.reasoning_network(values_output)
        
        # Apply moral attention
        if context is not None:
            attended_output, attention_weights = self.moral_attention(
                reasoning_output.unsqueeze(0),
                context.unsqueeze(0),
                context.unsqueeze(0)
            )
            attended_output = attended_output.squeeze(0)
        else:
            attended_output = reasoning_output
            attention_weights = None
            
        # Make decision
        decision_input = torch.cat([
            attended_output,
            context if context is not None else torch.zeros_like(attended_output)
        ], dim=-1)
        
        decision_logits = self.decision_network(decision_input)
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        # Record decision
        self.decision_history.append({
            'value_activations': {
                name: float(act) for name, act in zip(self.moral_values.keys(), value_activations[0])
            },
            'decision_probs': decision_probs[0].tolist(),
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'value_activations': value_activations,
            'reasoning_output': reasoning_output,
            'decision_logits': decision_logits,
            'decision_probs': decision_probs,
            'attention_weights': attention_weights
        }
        
    def get_active_values(self, threshold: float = 0.5) -> List[str]:
        """Get list of currently active moral values"""
        return [
            name for name, value in self.moral_values.items()
            if value.activation > threshold
        ]
        
    def get_decision_metrics(self) -> Dict[str, Any]:
        """Get metrics about moral decision making"""
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'average_activations': {},
                'latest_decision': None
            }
            
        recent_history = self.decision_history[-100:]  # Last 100 decisions
        
        # Calculate average value activations
        value_sums = {name: 0.0 for name in self.moral_values.keys()}
        for decision in recent_history:
            for name, activation in decision['value_activations'].items():
                value_sums[name] += activation
                
        avg_activations = {
            name: value_sum / len(recent_history)
            for name, value_sum in value_sums.items()
        }
        
        return {
            'total_decisions': len(self.decision_history),
            'average_activations': avg_activations,
            'latest_decision': self.decision_history[-1]
        }