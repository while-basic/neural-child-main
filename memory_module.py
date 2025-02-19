import torch
from torch import nn
from collections import deque
import random
import time
import math
from typing import Dict, List, Tuple
from replay_system import ReplayOptimizer

class MemoryCluster:
    """Represents a cluster of related memories"""
    def __init__(self, centroid: torch.Tensor):
        self.centroid = centroid
        self.memories = []
        self.importance = 1.0
        self.last_accessed = time.time()
        
    def add_memory(self, memory: torch.Tensor):
        self.memories.append(memory)
        self.centroid = torch.mean(torch.stack([m[0] for m in self.memories]), dim=0)
        
    def get_age(self) -> float:
        return time.time() - self.last_accessed

class DifferentiableMemory:
    def __init__(self, device='cuda'):
        self.device = device
        self.experiences = []
        self.encoder = torch.nn.Linear(768, 256).to(device)  # Embedding size to hidden size
        self.consolidation_network = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        ).to(device)
        
    def record_experience(self, input_tensor, internal_state, reward, timestamp, emotional_state):
        """Record an experience with proper tensor handling"""
        # Ensure input tensor is on correct device and has proper shape
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Ensure internal state is on correct device and has proper shape
        internal_state = internal_state.to(self.device)
        if internal_state.dim() == 1:
            internal_state = internal_state.unsqueeze(0)
            
        # Store experience
        self.experiences.append({
            'input': input_tensor.detach(),  # Detach to prevent memory leaks
            'internal_state': internal_state.detach(),
            'reward': float(reward),
            'timestamp': timestamp,
            'emotional_state': emotional_state.detach()
        })
        
    def sample_batch(self, batch_size):
        """Sample a batch of experiences"""
        if len(self.experiences) < batch_size:
            return None
            
        # Randomly sample experiences
        batch = random.sample(self.experiences, batch_size)
        return batch
        
    def replay_consolidation(self, batch_size=16):
        """Consolidate memories with proper tensor handling"""
        if not self.experiences:
            return 0.0
            
        try:
            # Sample batch
            batch = self.sample_batch(batch_size)
            if batch is None:
                return 0.0
                
            # Process batch
            inputs = []
            states = []
            
            for exp in batch:
                input_tensor = exp['input']
                state_tensor = exp['internal_state']
                
                # Ensure proper dimensions
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                    
                # Encode input
                encoded = self.encoder(input_tensor)
                
                inputs.append(encoded)
                states.append(state_tensor)
                
            # Stack tensors
            input_tensor = torch.cat(inputs, dim=0)
            state_tensor = torch.cat(states, dim=0)
            
            # Consolidate memories
            consolidated = self.consolidation_network(input_tensor)
            
            # Calculate loss
            criterion = torch.nn.MSELoss()
            loss = criterion(consolidated, state_tensor)
            
            # Update networks
            loss.backward()
            
            # Clean up old memories if needed
            if len(self.experiences) > 1000:  # Limit memory size
                self.experiences = self.experiences[-1000:]
                
            return loss.item()
            
        except Exception as e:
            print(f"Error in replay_consolidation: {str(e)}")
            return 0.0

    def compute_memory_importance(self, memory_embedding: torch.Tensor, 
                                emotional_state: torch.Tensor) -> float:
        # Ensure both tensors are on the same device
        memory_embedding = memory_embedding.to(self.device)
        emotional_state = emotional_state.to(self.device)
        
        # Ensure memory_embedding has the correct shape (take only the input part)
        if memory_embedding.dim() > 1:
            memory_embedding = memory_embedding.squeeze(0)
        memory_embedding = memory_embedding[:768]  # Take only the input part
        
        # Ensure emotional_state has the correct shape
        if emotional_state.dim() > 1:
            emotional_state = emotional_state.squeeze(0)
        
        # Combine the tensors
        combined = torch.cat([memory_embedding, emotional_state])
        
        # Add batch dimension for the network
        combined = combined.unsqueeze(0)
        
        importance = self.importance_net(combined)
        emotional_weight = torch.sum(emotional_state * self.emotional_importance)
        return importance.item() * emotional_weight.item()
    
    def find_similar_cluster(self, memory_embedding: torch.Tensor) -> Tuple[MemoryCluster, float]:
        if not self.long_term_clusters:
            return None, 0
        similarities = [torch.cosine_similarity(memory_embedding, cluster.centroid, dim=0)
                        for cluster in self.long_term_clusters]
        max_sim, idx = max((s, i) for i, s in enumerate(similarities))
        return self.long_term_clusters[idx], max_sim.item()
    
    def consolidate_memory(self, memory: torch.Tensor, importance: float):
        encoded_memory = self.encoder(memory)
        cluster, similarity = self.find_similar_cluster(encoded_memory)
        if similarity > self.consolidation_threshold:
            cluster.add_memory((encoded_memory, importance))
            cluster.importance = max(cluster.importance, importance)
            cluster.last_accessed = time.time()
        else:
            if len(self.long_term_clusters) < self.max_clusters:
                new_cluster = MemoryCluster(encoded_memory)
                new_cluster.add_memory((encoded_memory, importance))
                self.long_term_clusters.append(new_cluster)
            else:
                least_important = min(self.long_term_clusters, key=lambda c: c.importance * math.exp(-c.get_age() / 86400))
                self.long_term_clusters.remove(least_important)
                new_cluster = MemoryCluster(encoded_memory)
                new_cluster.add_memory((encoded_memory, importance))
                self.long_term_clusters.append(new_cluster)

    def retrieve_memories(self, cue: torch.Tensor, top_k: int = 5) -> List[torch.Tensor]:
        if not self.long_term_clusters:
            return []
        encoded_cue = self.encoder(cue)
        similarities = [torch.cosine_similarity(encoded_cue, cluster.centroid, dim=0)
                        for cluster in self.long_term_clusters]
        top_indices = torch.topk(torch.stack(similarities), top_k).indices
        retrieved_memories = []
        for idx in top_indices:
            retrieved_memories.extend([mem[0] for mem in self.long_term_clusters[idx].memories])
        return retrieved_memories[:top_k]
