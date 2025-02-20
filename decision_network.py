"""
Neural Network for Decision Making in Child Development AI
Created by Christopher Celaya

This module implements a decision-making neural network that considers:
- Conversation context through LSTM and attention mechanisms
- Emotional state integration
- Memory context integration
- Developmental stage adaptation
- Learning from feedback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from datetime import datetime

class ConversationEncoder(nn.Module):
    """
    Encodes conversation history using LSTM and attention mechanisms
    """
    def __init__(self, input_dim=384, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for processing conversation sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass through conversation encoder
        
        Args:
            x (torch.Tensor): Conversation embeddings [batch_size, seq_len, input_dim]
            mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            tuple: (encoded_sequence, attention_weights)
        """
        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply attention
        attended, attention_weights = self.attention(
            lstm_out.transpose(0, 1),  # [seq_len, batch_size, hidden_dim]
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            key_padding_mask=mask
        )
        
        # Transpose back and apply layer norm
        attended = attended.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        encoded = self.layer_norm(attended + lstm_out)  # Residual connection
        
        return encoded, attention_weights

class DecisionNetwork(nn.Module):
    """
    Neural network for making decisions based on conversation context,
    emotional state, memory context, and developmental stage
    """
    def __init__(
        self,
        input_dim=384,
        hidden_dim=256,
        num_actions=4,
        memory_size=1000,
        learning_rate=0.01
    ):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Conversation encoder
        self.conversation_encoder = ConversationEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # Emotional state processing
        self.emotion_processor = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Memory context processing
        self.memory_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Stage embedding processing
        self.stage_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Decision layers
        self.decision_layers = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, num_actions)
        
        # Confidence head
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Decision history
        self.decision_history = deque(maxlen=memory_size)
        self.confidence_history = deque(maxlen=memory_size)
        self.reward_history = deque(maxlen=memory_size)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(
        self,
        conversation_embeddings,
        emotional_state,
        memory_context,
        stage_embedding=None,
        mask=None
    ):
        """
        Forward pass through decision network
        
        Args:
            conversation_embeddings (torch.Tensor): [batch_size, seq_len, input_dim]
            emotional_state (torch.Tensor): [batch_size, 4]
            memory_context (torch.Tensor): [batch_size, input_dim]
            stage_embedding (torch.Tensor): [batch_size, input_dim//2]
            mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            dict: Network outputs including decision features and probabilities
        """
        # Ensure inputs are on the correct device
        conversation_embeddings = conversation_embeddings.to(self.device)
        emotional_state = emotional_state.to(self.device)
        memory_context = memory_context.to(self.device)
        if stage_embedding is not None:
            stage_embedding = stage_embedding.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        # Process conversation
        conv_encoded, attention_weights = self.conversation_encoder(
            conversation_embeddings,
            mask
        )
        conv_features = conv_encoded.mean(dim=1)  # Pool across sequence
        
        # Process emotional state
        emotion_features = self.emotion_processor(emotional_state)
        
        # Process memory context
        memory_features = self.memory_processor(memory_context)
        
        # Process stage embedding if provided
        if stage_embedding is not None:
            stage_features = self.stage_processor(stage_embedding)
        else:
            stage_features = torch.zeros_like(memory_features)
        
        # Combine features
        combined_features = torch.cat([
            conv_features,
            emotion_features,
            memory_features,
            stage_features
        ], dim=1)
        
        # Generate decision features
        decision_features = self.decision_layers(combined_features)
        
        # Generate action probabilities and confidence
        action_logits = self.action_head(decision_features)
        action_probs = F.softmax(action_logits, dim=-1)
        confidence = torch.sigmoid(self.confidence_head(decision_features))
        
        # Store decision for history
        self._store_decision(decision_features, confidence)
        
        return {
            'decision_features': decision_features,
            'action_probabilities': action_probs,
            'confidence': confidence,
            'attention_weights': attention_weights
        }
    
    def _store_decision(self, decision_features, confidence):
        """Store decision in history"""
        self.decision_history.append(decision_features.detach())
        self.confidence_history.append(confidence.detach())
    
    def update_from_feedback(self, reward):
        """
        Update network based on feedback
        
        Args:
            reward (float): Reward value between 0 and 1
        """
        if len(self.decision_history) > 0:
            self.reward_history.append(reward)
            
            # Get latest decision
            latest_decision = self.decision_history[-1]
            latest_confidence = self.confidence_history[-1]
            
            # Convert reward to tensor and ensure it requires grad
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=self.device, requires_grad=True)
            
            # Compute loss with stronger gradient
            confidence_loss = F.mse_loss(latest_confidence, reward_tensor) * 2.0
            
            # Update network
            self.optimizer.zero_grad()
            confidence_loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
    
    def get_decision_metrics(self):
        """
        Calculate metrics about decision-making performance
        
        Returns:
            dict: Various metrics about decision-making
        """
        if len(self.decision_history) == 0:
            return {
                'average_confidence': 0.0,
                'decision_stability': 0.0,
                'action_entropy': 0.0
            }
        
        # Calculate average confidence
        confidences = torch.stack(list(self.confidence_history))
        avg_confidence = confidences.mean().item()
        
        # Calculate decision stability (cosine similarity between consecutive decisions)
        decisions = torch.stack(list(self.decision_history))
        if len(decisions) > 1:
            similarities = F.cosine_similarity(decisions[:-1], decisions[1:])
            stability = similarities.mean().item()
        else:
            stability = 1.0
        
        # Calculate action entropy (diversity of decisions)
        if len(decisions) > 1:
            decision_distances = torch.cdist(decisions, decisions)
            entropy = torch.log(decision_distances.mean() + 1).item()
        else:
            entropy = 0.0
        
        return {
            'average_confidence': avg_confidence,
            'decision_stability': stability,
            'action_entropy': entropy
        }
    
    def save_state(self, path):
        """Save network state"""
        # Save all random states
        rng_states = {
            'python': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        
        # Save model state
        model_state = {
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'decision_history': [d.cpu().detach() for d in self.decision_history],
            'confidence_history': [c.cpu().detach() for c in self.confidence_history],
            'reward_history': list(self.reward_history),
            'rng_states': rng_states,
            'device': str(self.device)
        }
        
        torch.save(model_state, path, pickle_protocol=4)
    
    def load_state(self, path):
        """Load network state"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Restore random states
        if 'rng_states' in checkpoint:
            np.random.set_state(checkpoint['rng_states']['python'])
            torch.set_rng_state(checkpoint['rng_states']['torch'])
            if checkpoint['rng_states']['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda'])
        
        # Load model state
        self.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Ensure optimizer state is on correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        # Load history with proper device placement
        self.decision_history = deque(
            [d.to(self.device).detach() for d in checkpoint['decision_history']],
            maxlen=self.decision_history.maxlen
        )
        self.confidence_history = deque(
            [c.to(self.device).detach() for c in checkpoint['confidence_history']],
            maxlen=self.confidence_history.maxlen
        )
        self.reward_history = deque(
            checkpoint['reward_history'],
            maxlen=self.reward_history.maxlen
        ) 