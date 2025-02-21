import torch
from torch import nn
from collections import deque
import random
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from replay_system import ReplayOptimizer
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np

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

class MemoryStore:
    def __init__(self, persist_directory="memories"):
        self.persist_directory = persist_directory
        self.memories = {
            'semantic': [],
            'episodic': [],
            'emotional': []
        }
        self.memory_index = defaultdict(list)
        os.makedirs(persist_directory, exist_ok=True)
        self.load_memories()
    
    def store_semantic_memory(self, content, embedding, metadata=None):
        """Store a semantic memory with its embedding"""
        memory = {
            'id': len(self.memories['semantic']),
            'content': content,
            'embedding': embedding,
            'type': 'semantic',
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.memories['semantic'].append(memory)
        self._update_index(memory)
        self._save_memories()
        return memory['id']
    
    def store_episodic_memory(self, content, embedding, metadata=None, emotional_state=None):
        """Store an episodic memory with its embedding and optional emotional state"""
        memory = {
            'id': len(self.memories['episodic']),
            'content': content,
            'embedding': embedding,
            'type': 'episodic',
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'emotional_state': emotional_state or {}
        }
        self.memories['episodic'].append(memory)
        self._update_index(memory)
        self._save_memories()
        return memory['id']
    
    def store_emotional_memory(self, content, embedding, emotional_state, metadata=None):
        """Store an emotional memory with its embedding and emotional state"""
        memory = {
            'id': len(self.memories['emotional']),
            'content': content,
            'embedding': embedding,
            'type': 'emotional',
            'emotional_state': emotional_state,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.memories['emotional'].append(memory)
        self._update_index(memory)
        self._save_memories()
        return memory['id']
    
    def query_memories(self, query_embedding, n_results=3, memory_type=None):
        """Query memories using cosine similarity"""
        query_embedding = np.array(query_embedding)
        results = []
        
        # Determine which memory types to search
        memory_types = [memory_type] if memory_type else self.memories.keys()
        
        for mem_type in memory_types:
            for memory in self.memories[mem_type]:
                memory_embedding = np.array(memory['embedding'])
                similarity = np.dot(query_embedding, memory_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                )
                results.append((similarity, memory))
        
        # Sort by similarity and return top n results
        results.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in results[:n_results]]
    
    def get_memory_stats(self):
        """Get statistics about stored memories"""
        return {
            'total_memories': sum(len(memories) for memories in self.memories.values()),
            'semantic_count': len(self.memories['semantic']),
            'episodic_count': len(self.memories['episodic']),
            'emotional_count': len(self.memories['emotional'])
        }
    
    def _update_index(self, memory):
        """Update memory index with new memory"""
        words = memory['content'].lower().split()
        for word in words:
            self.memory_index[word].append(memory['id'])
    
    def _save_memories(self):
        """Save memories to disk"""
        try:
            with open(os.path.join(self.persist_directory, 'memories.json'), 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            print(f"Error saving memories: {str(e)}")
    
    def load_memories(self):
        """Load memories from disk"""
        try:
            memory_file = os.path.join(self.persist_directory, 'memories.json')
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    self.memories = json.load(f)
                # Rebuild index
                self.memory_index.clear()
                for memory_type in self.memories.values():
                    for memory in memory_type:
                        self._update_index(memory)
        except Exception as e:
            print(f"Error loading memories: {str(e)}")

class DifferentiableMemory:
    def __init__(self, device='cpu'):
        self.device = device
        self.experiences = []
        self.encoder = torch.nn.Linear(768, 256).to(device)
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
            
        # Ensure emotional state is on correct device
        emotional_state = emotional_state.to(self.device)
        
        # Store experience
        self.experiences.append({
            'input': input_tensor.detach(),
            'internal_state': internal_state.detach(),
            'reward': float(reward),
            'timestamp': timestamp,
            'emotional_state': emotional_state.detach()
        })
        
        # Limit memory size
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-1000:]
        
    def sample_batch(self, batch_size):
        """Sample a batch of experiences with proper device handling"""
        if len(self.experiences) < batch_size:
            return None
            
        # Randomly sample experiences
        batch = random.sample(self.experiences, batch_size)
        
        # Prepare tensors for batch processing
        inputs = []
        states = []
        rewards = []
        emotional_states = []
        
        for exp in batch:
            # Move tensors to correct device
            inputs.append(exp['input'].to(self.device))
            states.append(exp['internal_state'].to(self.device))
            rewards.append(torch.tensor([exp['reward']], device=self.device))
            emotional_states.append(exp['emotional_state'].to(self.device))
        
        # Stack tensors
        batch_data = {
            'inputs': torch.cat(inputs, dim=0),
            'states': torch.cat(states, dim=0),
            'rewards': torch.stack(rewards),
            'emotional_states': torch.stack(emotional_states)
        }
        
        return batch_data
        
    def replay_consolidation(self, batch_size=16):
        """Consolidate memories with proper tensor handling"""
        if not self.experiences:
            return 0.0
            
        try:
            # Sample batch
            batch = self.sample_batch(batch_size)
            if batch is None:
                return 0.0
                
            # Process inputs through encoder
            encoded = self.encoder(batch['inputs'])
            
            # Consolidate memories
            consolidated = self.consolidation_network(encoded)
            
            # Calculate loss with emotional weighting
            criterion = torch.nn.MSELoss(reduction='none')
            base_loss = criterion(consolidated, batch['states'])
            
            # Weight loss by emotional significance
            emotional_intensity = torch.mean(batch['emotional_states'], dim=1)
            weighted_loss = base_loss * emotional_intensity.unsqueeze(1)
            
            # Calculate final loss
            loss = torch.mean(weighted_loss)
            
            # Update networks
            loss.backward()
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in replay_consolidation: {str(e)}")
            return 0.0

    def to(self, device):
        """Move memory module to specified device"""
        self.device = device
        self.encoder = self.encoder.to(device)
        self.consolidation_network = self.consolidation_network.to(device)
        return self

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

class MemoryModule(nn.Module):
    """Neural network module for memory management"""
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Working memory
        self.working_memory = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Long-term memory
        self.long_term_memory = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Memory consolidation
        self.consolidation_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Memory retrieval
        self.retrieval_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Initialize states
        self.reset_states()
        
        # Memory storage
        self.memory_buffer = []
        self.memory_timestamps = []
        
    def reset_states(self):
        """Reset memory states"""
        self.working_memory_state = None
        self.long_term_state = None
        
    def forward(self, 
                input_data: torch.Tensor,
                memory_query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Process input through memory systems"""
        batch_size = input_data.shape[0]
        device = input_data.device
        
        # Process through working memory
        working_output, self.working_memory_state = self.working_memory(
            input_data.unsqueeze(1),
            self.working_memory_state
        )
        working_output = working_output.squeeze(1)
        
        # Process through long-term memory
        long_term_output, self.long_term_state = self.long_term_memory(
            working_output.unsqueeze(1),
            self.long_term_state
        )
        long_term_output = long_term_output.squeeze(1)
        
        # Consolidate memories
        consolidated_memory = self.consolidation_network(
            torch.cat([working_output, long_term_output], dim=-1)
        )
        
        # Store memory
        self.store_memory(consolidated_memory)
        
        # Retrieve relevant memories
        if memory_query is None:
            memory_query = consolidated_memory
            
        retrieved_memory = self.retrieve_memories(memory_query)
        
        return {
            'working_memory': working_output,
            'long_term_memory': long_term_output,
            'consolidated_memory': consolidated_memory,
            'retrieved_memory': retrieved_memory
        }
        
    def store_memory(self, memory: torch.Tensor):
        """Store memory in buffer"""
        # Convert to CPU for storage
        memory_cpu = memory.detach().cpu()
        
        # Add to buffer with timestamp
        self.memory_buffer.append(memory_cpu)
        self.memory_timestamps.append(datetime.now())
        
        # Keep only recent memories (last 100)
        if len(self.memory_buffer) > 100:
            self.memory_buffer.pop(0)
            self.memory_timestamps.pop(0)
            
    def retrieve_memories(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve relevant memories using attention"""
        if not self.memory_buffer:
            return query
            
        # Stack memories
        memories = torch.stack(self.memory_buffer).to(query.device)
        
        # Apply attention
        retrieved, _ = self.retrieval_attention(
            query.unsqueeze(0),
            memories.unsqueeze(0),
            memories.unsqueeze(0)
        )
        
        return retrieved.squeeze(0)
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        return {
            'total_memories': len(self.memory_buffer),
            'oldest_memory': self.memory_timestamps[0].isoformat() if self.memory_timestamps else None,
            'newest_memory': self.memory_timestamps[-1].isoformat() if self.memory_timestamps else None
        }
