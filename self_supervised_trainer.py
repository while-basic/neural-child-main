# self_supervised_trainer.py

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import time

class AutonomousTrainer:
    def __init__(self, brain, memory, morality, batch_size=32):
        self.brain = brain
        self.memory = memory
        self.morality = morality
        self.batch_size = batch_size
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.001)

    def training_step(self, input_tensor):
        """Perform a training step with proper tensor handling"""
        self.optimizer.zero_grad()
        
        # Ensure input tensor is properly shaped
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Get prediction from brain
        prediction = self.brain(input_tensor)
        
        # Get replay batch from memory
        replay_batch = self.memory.sample_batch(self.batch_size)
        if replay_batch is None or len(replay_batch) == 0:
            return torch.tensor(0.0, device='cuda')
            
        # Process replay batch
        replay_loss = self._process_batch(replay_batch)
        
        # Combine losses
        total_loss = replay_loss
        
        # Only backward if we have a valid tensor
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()
            return total_loss.item()
        
        return 0.0
        
    def _process_batch(self, batch):
        """Process a batch of experiences with proper tensor handling"""
        try:
            inputs = []
            targets = []
            
            for experience in batch:
                if isinstance(experience, dict):
                    input_data = experience.get('input')
                    target_data = experience.get('target')
                    
                    if isinstance(input_data, torch.Tensor) and isinstance(target_data, torch.Tensor):
                        # Ensure proper dimensions
                        if input_data.dim() == 1:
                            input_data = input_data.unsqueeze(0)
                        if target_data.dim() == 1:
                            target_data = target_data.unsqueeze(0)
                            
                        inputs.append(input_data)
                        targets.append(target_data)
            
            if not inputs or not targets:
                return torch.tensor(0.0, device='cuda', requires_grad=True)
                
            # Stack tensors
            input_tensor = torch.cat(inputs, dim=0)
            target_tensor = torch.cat(targets, dim=0)
            
            # Get prediction
            prediction = self.brain(input_tensor)
            
            # Calculate loss
            loss = self.criterion(prediction, target_tensor)
            return loss
            
        except Exception as e:
            print(f"Error in _process_batch: {str(e)}")
            return torch.tensor(0.0, device='cuda', requires_grad=True)

    def _safe_tensor_op(self, tensor, requires_grad=False):
        """Helper method for safe tensor operations"""
        result = tensor.clone().detach()
        if requires_grad:
            result.requires_grad_(True)
        return result.to(self.brain.device)