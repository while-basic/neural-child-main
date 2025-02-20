# self_awareness_network.py
# Description: Neural network for self-awareness and metacognition in child development
# Created by: Christopher Celaya

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import networkx as nx
from enum import Enum, auto
import json

class SelfAwarenessLevel(Enum):
    """Levels of self-awareness development"""
    PHYSICAL = auto()      # Basic physical self-awareness (0-18 months)
    MIRROR = auto()        # Mirror self-recognition (18-24 months)
    EMOTIONAL = auto()     # Emotional self-awareness (2-3 years)
    COGNITIVE = auto()     # Understanding own thoughts (3-4 years)
    METACOGNITIVE = auto() # Thinking about thinking (4-5 years)
    SOCIAL = auto()        # Social self-awareness (5+ years)
    ABSTRACT = auto()      # Abstract self-concept (adolescence)

class SelfAwarenessNetwork(nn.Module):
    """Neural network for developing and tracking self-awareness"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature dimensions
        self.physical_dim = hidden_dim // 4
        self.emotional_dim = hidden_dim // 4  
        self.cognitive_dim = hidden_dim // 4
        self.attention_dim = hidden_dim // 4
        self.processed_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(384, hidden_dim)
        
        # Feature networks
        self.physical_network = nn.Sequential(
            nn.Linear(hidden_dim, self.physical_dim),
            nn.ReLU(),
            nn.Linear(self.physical_dim, self.physical_dim)
        )
        
        self.emotional_network = nn.Sequential(
            nn.Linear(hidden_dim, self.emotional_dim),
            nn.ReLU(),
            nn.Linear(self.emotional_dim, self.emotional_dim)
        )
        
        self.cognitive_network = nn.Sequential(
            nn.Linear(hidden_dim, self.cognitive_dim),
            nn.ReLU(),
            nn.Linear(self.cognitive_dim, self.cognitive_dim)
        )
        
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_dim, self.attention_dim),
            nn.ReLU(),
            nn.Linear(self.attention_dim, self.attention_dim)
        )
        
        # Feature compression
        self.memory_compression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Emotional memory network
        self.emotional_memory = nn.LSTM(
            input_size=hidden_dim // 2 + 4,  # Compressed features + emotions
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Word learning network
        self.word_learning = nn.Sequential(
            nn.Linear(hidden_dim + 384, hidden_dim),  # Word embedding dim = 384
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Emotional association network
        self.emotional_association = nn.Sequential(
            nn.Linear(hidden_dim + 384, hidden_dim),  # Word embedding dim = 384
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 emotion dimensions
            nn.Sigmoid()
        )
        
        # Initialize states
        self.emotional_state = torch.zeros(1, 4)  # [joy, trust, fear, surprise]
        self.attention_state = torch.zeros(1, self.attention_dim)
        self.metacognition = 0.0
        
        # Storage
        self.emotional_memories = []  # List of (timestamp, features, emotions)
        self.known_words = set()
        self.word_emotions = {}  # word -> emotion vector mapping
        
        # Self-concept graph
        self.self_concept_graph = nx.Graph()
        
        # Development tracking
        self.current_level = SelfAwarenessLevel.PHYSICAL
        self.level_progress = 0.0
        self.reflection_history = []
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process input through the network"""
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Process features
        physical_features = self.physical_network(x)
        emotional_features = self.emotional_network(x) 
        cognitive_features = self.cognitive_network(x)
        attention_features = self.attention_network(x)
        
        # Combine features
        combined_features = torch.cat([
            physical_features,
            emotional_features, 
            cognitive_features,
            attention_features
        ], dim=1)
        
        # Compress features for memory
        compressed_features = self.memory_compression(x)  # Use projected input
        
        # Update emotional memory
        if len(self.emotional_memories) > 0:
            memory_input = torch.cat([
                compressed_features,
                self.emotional_state.expand(batch_size, -1)
            ], dim=1).unsqueeze(1)  # Add sequence dimension
            
            memory_output, _ = self.emotional_memory(memory_input)
            memory_output = memory_output.squeeze(1)
        else:
            memory_output = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            
        # Calculate metacognition
        self.metacognition = torch.mean(attention_features).item()
        
        # Update emotional state based on attention and memory
        emotional_update = torch.sigmoid(memory_output[:, :4])  # First 4 dimensions for emotions
        self.emotional_state = 0.9 * self.emotional_state + 0.1 * emotional_update
        
        return combined_features, {
            'physical_features': physical_features,
            'emotional_features': emotional_features,
            'cognitive_features': cognitive_features,
            'attention_features': attention_features,
            'combined_features': combined_features,
            'compressed_features': compressed_features,
            'memory_output': memory_output,
            'metacognition': self.metacognition
        }
    
    def update_self_concept(self, interaction: Dict):
        """Update self-concept graph with new interaction"""
        # Add new node for current interaction
        node_id = len(self.self_concept_graph.nodes)
        self.self_concept_graph.add_node(
            node_id,
            content=interaction['content'],
            emotions=interaction['emotional_state']
        )
        
        # Connect to similar nodes
        for other_id, other_data in self.self_concept_graph.nodes(data=True):
            if other_id != node_id:
                # Calculate emotional similarity
                current_emotions = torch.tensor([
                    interaction['emotional_state']['joy'],
                    interaction['emotional_state']['trust'],
                    interaction['emotional_state']['fear'],
                    interaction['emotional_state']['surprise']
                ])
                
                other_emotions = torch.tensor([
                    other_data['emotions']['joy'],
                    other_data['emotions']['trust'],
                    other_data['emotions']['fear'],
                    other_data['emotions']['surprise']
                ])
                
                similarity = F.cosine_similarity(
                    current_emotions.unsqueeze(0),
                    other_emotions.unsqueeze(0)
                ).item()
                
                # Add edge if similarity is high enough
                if similarity > 0.3:  # Lowered threshold
                    self.self_concept_graph.add_edge(
                        node_id,
                        other_id,
                        weight=similarity
                    )
                    
    def get_development_metrics(self) -> Dict:
        """Get current development metrics"""
        return {
            'self_concept_size': len(self.self_concept_graph.nodes()),
            'self_concept_connections': len(self.self_concept_graph.edges()),
            'known_words': len(self.known_words),
            'emotional_memories': len(self.emotional_memories),
            'metacognition': self.metacognition
        }
    
    def save_state(self, path: str):
        """Save network state"""
        state = {
            'emotional_state': self.emotional_state.tolist(),
            'attention_state': self.attention_state.tolist(),
            'metacognition': self.metacognition,
            'known_words': list(self.known_words),
            'word_emotions': {word: tensor.tolist() for word, tensor in self.word_emotions.items()},
            'emotional_memories': self.emotional_memories,
            'self_concept_graph': {
                'nodes': list(self.self_concept_graph.nodes(data=True)),
                'edges': list(self.self_concept_graph.edges(data=True))
            }
        }
        torch.save(state, path)
    
    def load_state(self, path: str):
        """Load network state"""
        state = torch.load(path)
        self.emotional_state = torch.tensor(state['emotional_state'])
        self.attention_state = torch.tensor(state['attention_state'])
        self.metacognition = state['metacognition']
        self.known_words = set(state['known_words'])
        self.word_emotions = {word: torch.tensor(data) for word, data in state['word_emotions'].items()}
        self.emotional_memories = state['emotional_memories']
        
        # Rebuild graph
        self.self_concept_graph = nx.Graph()
        for node, data in state['self_concept_graph']['nodes']:
            self.self_concept_graph.add_node(node, **data)
        for u, v, data in state['self_concept_graph']['edges']:
            self.self_concept_graph.add_edge(u, v, **data)
    
    def process_emotional_memory(self, features: torch.Tensor) -> Optional[Dict]:
        """Process and potentially recall emotional memory"""
        if len(self.emotional_memories) == 0:
            return None
            
        # Compress current features
        compressed = self.memory_compression(features)
        
        # Calculate similarity with stored memories
        max_similarity = -1
        closest_memory = None
        
        for timestamp, mem_features, mem_emotions in self.emotional_memories:
            similarity = F.cosine_similarity(
                compressed,
                mem_features.unsqueeze(0)
            ).item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                closest_memory = {
                    'timestamp': timestamp,
                    'features': mem_features,
                    'emotions': mem_emotions,
                    'similarity': similarity
                }
                
        # Only recall if similarity is high enough
        if max_similarity > 0.7:
            return closest_memory
        return None
        
    def learn_word(self, word: str, embedding: torch.Tensor, features: torch.Tensor) -> Dict:
        """Learn a new word and its emotional associations"""
        # Ensure embeddings have batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        # Combine features with word embedding
        combined = torch.cat([features, embedding], dim=1)
        
        # Calculate learning confidence
        learning_confidence = self.word_learning(combined)
        
        # Get emotional association if confidence is high enough
        result = {
            'word': word,
            'learning_confidence': float(learning_confidence.squeeze().item())
        }
        
        if result['learning_confidence'] > 0.5:
            emotional_assoc = self.emotional_association(combined)
            self.known_words.add(word)
            self.word_emotions[word] = emotional_assoc.detach()
            result['emotional_association'] = emotional_assoc.squeeze().tolist()
            
        return result
        
    def recall_emotional_memory(self, query_features: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Recall most relevant emotional memory"""
        if not self.emotional_memories:
            return None
            
        # Compress query features
        compressed_query = self.memory_compression(query_features)
            
        # Calculate similarity with all memories
        similarities = []
        for memory in self.emotional_memories:
            sim = F.cosine_similarity(
                compressed_query,
                memory['features'],
                dim=-1
            )
            similarities.append(sim)
            
        # Get most similar memory
        max_idx = torch.argmax(torch.tensor(similarities))
        memory = self.emotional_memories[max_idx]
        
        return {
            'emotions': memory['emotions'].tolist(),
            'association': memory['association'].tolist(),
            'timestamp': memory['timestamp']
        }
        
    def get_word_emotion(self, word: str) -> Optional[Dict[str, Any]]:
        """Get emotional association for a word"""
        word = word.lower()
        if word in self.word_emotions:
            emotion_data = self.word_emotions[word]
            return {
                'word': word,
                'emotions': emotion_data.tolist()
            }
        return None 