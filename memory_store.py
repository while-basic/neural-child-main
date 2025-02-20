# memory_store.py
# Description: Memory store for neural child development
# Created by: Christopher Celaya

import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sentence_transformers import SentenceTransformer
from logger import DevelopmentLogger

class MemoryStore:
    def __init__(self, logger: Optional[DevelopmentLogger] = None, persist_directory: str = "memories", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the memory store with a sentence transformer model."""
        self.logger = logger or DevelopmentLogger()
        self.persist_directory = persist_directory
        self.model = SentenceTransformer(model_name)
        self.memories = []
        self.emotional_memories = []
        self.memory_embeddings = []
        self.current_emotional_state = torch.zeros(4)  # [joy, trust, fear, surprise]
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize memory stats
        self.stats = {
            "total_memories": 0,
            "semantic_count": 0,
            "episodic_count": 0,
            "emotional_count": 0,
            "last_consolidated": None
        }
        
    def store_episodic_memory(self, content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Store an episodic memory (event-based experience)"""
        try:
            memory_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['episodic_count']}"
            
            memory = {
                'id': memory_id,
                'content': content,
                'embedding': embedding,
                'metadata': {
                    **metadata,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'episodic'
                }
            }
            
            self.memories.append(memory)
            self.memory_embeddings.append(torch.tensor(embedding))
            self.stats["episodic_count"] += 1
            self.stats["total_memories"] += 1
            
            return memory_id
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'store_episodic_memory',
                'content': content
            })
            return ""
    
    def store_semantic_memory(self, content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Store a semantic memory (factual knowledge)"""
        try:
            memory_id = f"sem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['semantic_count']}"
            
            memory = {
                'id': memory_id,
                'content': content,
                'embedding': embedding,
                'metadata': {
                    **metadata,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'semantic'
                }
            }
            
            self.memories.append(memory)
            self.memory_embeddings.append(torch.tensor(embedding))
            self.stats["semantic_count"] += 1
            self.stats["total_memories"] += 1
            
            return memory_id
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'store_semantic_memory',
                'content': content
            })
            return ""
    
    def store_emotional_memory(self, content: str, embedding: List[float], emotional_state: Dict[str, float], metadata: Dict[str, Any]) -> str:
        """Store an emotional memory with associated emotional state"""
        try:
            memory_id = f"em_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['emotional_count']}"
            
            memory = {
                'id': memory_id,
                'content': content,
                'embedding': embedding,
                'metadata': {
                    **metadata,
                    **emotional_state,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'emotional'
                }
            }
            
            self.emotional_memories.append(memory)
            self.memories.append(memory)
            self.memory_embeddings.append(torch.tensor(embedding))
            self.stats["emotional_count"] += 1
            self.stats["total_memories"] += 1
            
            return memory_id
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'store_emotional_memory',
                'content': content
            })
            return ""
    
    def query_memories(self, query_embedding: List[float], n_results: int = 5, memory_type: str = "all") -> List[Dict]:
        """Query memories based on vector similarity"""
        try:
            results = []
            query_embedding = torch.tensor(query_embedding)
            
            # Determine which memories to search
            search_memories = []
            if memory_type == "all" or memory_type == "episodic":
                search_memories.extend([m for m in self.memories if m['metadata']['type'] == 'episodic'])
            if memory_type == "all" or memory_type == "semantic":
                search_memories.extend([m for m in self.memories if m['metadata']['type'] == 'semantic'])
            if memory_type == "all" or memory_type == "emotional":
                search_memories.extend([m for m in self.memories if m['metadata']['type'] == 'emotional'])
            
            # Calculate similarities
            for memory in search_memories:
                memory_embedding = torch.tensor(memory['embedding'])
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    memory_embedding.unsqueeze(0)
                ).item()
                results.append((similarity, memory))
            
            # Sort by similarity and return top n results
            results.sort(key=lambda x: x[0], reverse=True)
            return [memory for _, memory in results[:n_results]]
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'query_memories',
                'memory_type': memory_type
            })
            return []
    
    def retrieve_similar_memories(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve k most similar memories to the query."""
        try:
            if not self.memories:
                return []
            
            # Encode query
            query_embedding = self.model.encode(query, convert_to_tensor=True).cpu()
            
            # Calculate similarities
            similarities = []
            for memory in self.memories:
                memory_embedding = torch.tensor(memory['embedding'])
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    memory_embedding.unsqueeze(0)
                ).item()
                similarities.append((similarity, memory))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [memory for _, memory in similarities[:k]]
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'retrieve_similar_memories',
                'query': query,
                'k': k
            })
            return []
    
    def get_emotional_memories(self, emotion: Optional[str] = None) -> List[Dict]:
        """Retrieve emotional memories, optionally filtered by emotion type."""
        try:
            if emotion:
                return [
                    memory for memory in self.emotional_memories
                    if memory['metadata'].get('emotion', '').lower() == emotion.lower()
                ]
            return self.emotional_memories
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'get_emotional_memories',
                'emotion': emotion
            })
            return []
    
    def update_emotional_state(self, impact: torch.Tensor) -> Dict:
        """Update the current emotional state based on new emotional impact."""
        try:
            # Apply emotional impact with decay
            decay = 0.9
            self.current_emotional_state = (
                self.current_emotional_state * decay + impact * (1 - decay)
            )
            
            # Normalize emotional state
            self.current_emotional_state = torch.clamp(
                self.current_emotional_state,
                min=0.0,
                max=1.0
            )
            
            return {
                'success': True,
                'emotional_state': self.current_emotional_state.cpu().tolist()
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'update_emotional_state',
                'impact': impact.cpu().tolist() if impact is not None else None
            })
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        try:
            if not self.memories:
                return {
                    'total_memories': 0,
                    'emotional_memories': 0,
                    'average_emotional_impact': None,
                    'emotion_distribution': {}
                }
            
            # Calculate statistics
            total_memories = len(self.memories)
            emotional_memories = len(self.emotional_memories)
            
            # Calculate average emotional impact
            emotional_impacts = [
                memory['emotional_impact']
                for memory in self.memories
                if memory['emotional_impact'] is not None
            ]
            
            average_emotional_impact = (
                np.mean(emotional_impacts, axis=0).tolist()
                if emotional_impacts else None
            )
            
            # Calculate emotion distribution
            emotion_distribution = {}
            for memory in self.emotional_memories:
                emotion = memory['metadata'].get('emotion', '')
                emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            return {
                'total_memories': total_memories,
                'emotional_memories': emotional_memories,
                'average_emotional_impact': average_emotional_impact,
                'emotion_distribution': emotion_distribution,
                'current_emotional_state': self.current_emotional_state.cpu().tolist()
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'get_memory_statistics'
            })
            return {
                'error': str(e)
            }
    
    def save_memories(self, filepath: str):
        """Save all memories to a file."""
        try:
            data = {
                'memories': self.memories,
                'emotional_memories': self.emotional_memories,
                'current_emotional_state': self.current_emotional_state.cpu().tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'save_memories',
                'filepath': filepath
            })
    
    def load_memories(self, filepath: str):
        """Load memories from a file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.memories = data.get('memories', [])
            self.emotional_memories = data.get('emotional_memories', [])
            self.current_emotional_state = torch.tensor(
                data.get('current_emotional_state', [0.0, 0.0, 0.0, 0.0])
            )
            
            # Rebuild memory embeddings
            self.memory_embeddings = [
                torch.tensor(memory['embedding'])
                for memory in self.memories
            ]
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'load_memories',
                'filepath': filepath
            })
    
    def clear_memories(self):
        """Clear all stored memories."""
        self.memories = []
        self.emotional_memories = []
        self.memory_embeddings = []
        self.current_emotional_state = torch.zeros(4)
        
    def get_emotional_state(self) -> torch.Tensor:
        """Get the current emotional state."""
        return self.current_emotional_state.clone()
        
    def get_memory_by_id(self, memory_id: int) -> Optional[Dict]:
        """Retrieve a specific memory by its ID."""
        try:
            if 0 <= memory_id < len(self.memories):
                return self.memories[memory_id]
            return None
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'get_memory_by_id',
                'memory_id': memory_id
            })
            return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return self.stats

    def save_state(self, filepath: str):
        """Save the current state of the memory store to a JSON file."""
        try:
            data = {
                'memories': self.memories,
                'emotional_memories': self.emotional_memories,
                'current_emotional_state': self.current_emotional_state.cpu().tolist(),
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'save_state',
                'filepath': filepath
            })

    def load_state(self, filepath: str):
        """Load the memory store state from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.memories = data.get('memories', [])
            self.emotional_memories = data.get('emotional_memories', [])
            self.current_emotional_state = torch.tensor(
                data.get('current_emotional_state', [0.0, 0.0, 0.0, 0.0])
            )
            self.stats = data.get('stats', {
                "total_memories": 0,
                "semantic_count": 0,
                "episodic_count": 0,
                "emotional_count": 0,
                "last_consolidated": None
            })
            
            # Rebuild memory embeddings
            self.memory_embeddings = [
                torch.tensor(memory['embedding'])
                for memory in self.memories
            ]
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'load_state',
                'filepath': filepath
            }) 