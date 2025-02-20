# self_supervised_trainer.py
# Description: Self-supervised learning trainer for neural child development
# Created by: Christopher Celaya

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from datetime import datetime

class SelfSupervisedTrainer:
    """Trainer for self-supervised learning tasks"""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.training_history = []
        
    def create_masked_input(self, input_data: torch.Tensor, mask_prob: float = 0.15) -> Dict[str, torch.Tensor]:
        """Create masked version of input for self-supervised learning"""
        mask = torch.rand_like(input_data) < mask_prob
        masked_input = input_data.clone()
        masked_input[mask] = 0
        
        return {
            'original': input_data,
            'masked': masked_input,
            'mask': mask
        }
        
    def train_step(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Perform one training step"""
        self.model.train()
        
        # Create masked input
        masked_data = self.create_masked_input(input_data)
        
        # Forward pass
        outputs = self.model(masked_data['masked'])
        
        # Calculate reconstruction loss
        recon_loss = F.mse_loss(
            outputs['consolidated_memory'],
            masked_data['original']
        )
        
        # Calculate contrastive loss if applicable
        if 'embeddings' in outputs:
            pos_pairs = outputs['embeddings']
            neg_pairs = torch.roll(pos_pairs, 1, dims=0)  # Create negative pairs
            contrastive_loss = F.cosine_embedding_loss(
                pos_pairs,
                neg_pairs,
                torch.ones(pos_pairs.size(0), device=pos_pairs.device)
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=input_data.device)
            
        # Total loss
        total_loss = recon_loss + 0.1 * contrastive_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Record metrics
        metrics = {
            'total_loss': float(total_loss.item()),
            'reconstruction_loss': float(recon_loss.item()),
            'contrastive_loss': float(contrastive_loss.item()),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(metrics)
        
        return metrics
        
    def evaluate(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        
        with torch.no_grad():
            # Create masked input
            masked_data = self.create_masked_input(input_data)
            
            # Forward pass
            outputs = self.model(masked_data['masked'])
            
            # Calculate metrics
            recon_loss = F.mse_loss(
                outputs['consolidated_memory'],
                masked_data['original']
            )
            
            if 'embeddings' in outputs:
                pos_pairs = outputs['embeddings']
                neg_pairs = torch.roll(pos_pairs, 1, dims=0)
                contrastive_loss = F.cosine_embedding_loss(
                    pos_pairs,
                    neg_pairs,
                    torch.ones(pos_pairs.size(0), device=pos_pairs.device)
                )
            else:
                contrastive_loss = torch.tensor(0.0, device=input_data.device)
                
        return {
            'reconstruction_loss': float(recon_loss.item()),
            'contrastive_loss': float(contrastive_loss.item())
        }
        
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.training_history:
            return {
                'total_steps': 0,
                'average_loss': 0.0,
                'latest_metrics': None
            }
            
        recent_history = self.training_history[-100:]  # Last 100 steps
        
        return {
            'total_steps': len(self.training_history),
            'average_loss': sum(h['total_loss'] for h in recent_history) / len(recent_history),
            'latest_metrics': self.training_history[-1]
        }