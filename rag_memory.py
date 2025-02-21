# rag_memory.py
# Description: Retrieval-Augmented Generation system for neural child memories
# Created by: Christopher Celaya

import torch
from typing import Dict, List, Optional, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from memory_context import MemoryContext

@dataclass
class MemoryContext:
    """Context information for memory retrieval"""
    query: str
    emotional_state: Dict[str, float]
    brain_state: Dict[str, Any]
    developmental_stage: str
    age_months: int
    timestamp: datetime

class RAGMemorySystem:
    """Advanced RAG memory system using ChromaDB for storing and retrieving memories with emotional context"""
    
    def __init__(self, 
                chat_interface,
                sentence_transformer_model: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG memory system"""
        # Use existing ChromaDB client from chat interface
        self.chroma_client = chat_interface.chroma_client
        
        # Use existing collections
        self.emotional_collection = chat_interface.emotional_memory_collection
        self.episodic_collection = chat_interface.episodic_memory_collection
        self.semantic_collection = chat_interface.semantic_memory_collection
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model)
        
        # Initialize stats
        self.stats = {
            "total_memories": 0,
            "semantic_memories": 0,
            "episodic_memories": 0,
            "emotional_memories": 0,
            "last_consolidated": None
        }

    def store_memory(self, 
                    content: str,
                    memory_type: str,
                    emotional_state: Dict[str, float],
                    brain_state: Dict[str, float],
                    metadata: Dict[str, Any]) -> str:
        """Store a memory with its associated states and metadata"""
        try:
            # Generate memory ID
            memory_id = f"{memory_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['total_memories']}"
            
            # Get embedding
            embedding = self.sentence_transformer.encode(content)
            
            # Flatten emotional and brain states into metadata
            flattened_metadata = {
                # Core metadata
                'type': memory_type,  # Store type directly in metadata
                'timestamp': datetime.now().isoformat(),
                'appropriate': True,
                
                # Emotional state
                'emotional_joy': float(emotional_state.get('joy', 0.0)),
                'emotional_trust': float(emotional_state.get('trust', 0.0)),
                'emotional_fear': float(emotional_state.get('fear', 0.0)),
                'emotional_surprise': float(emotional_state.get('surprise', 0.0)),
                
                # Brain state
                'brain_arousal': float(brain_state.get('arousal', 0.5)),
                'brain_attention': float(brain_state.get('attention', 0.5)),
                'brain_emotional_valence': float(brain_state.get('emotional_valence', 0.0)),
                'brain_consciousness': float(brain_state.get('consciousness', 1.0)),
                'brain_stress': float(brain_state.get('stress', 0.2)),
                'brain_fatigue': float(brain_state.get('fatigue', 0.0)),
            }
            
            # Add additional metadata
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    flattened_metadata[key] = value
            
            # Store in appropriate collection
            if memory_type == 'semantic':
                self.semantic_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[content],
                    metadatas=[flattened_metadata],
                    ids=[memory_id]
                )
                self.stats['semantic_memories'] += 1
            elif memory_type == 'episodic':
                self.episodic_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[content],
                    metadatas=[flattened_metadata],
                    ids=[memory_id]
                )
                self.stats['episodic_memories'] += 1
            else:  # emotional
                self.emotional_collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[content],
                    metadatas=[flattened_metadata],
                    ids=[memory_id]
                )
                self.stats['emotional_memories'] += 1
            
            self.stats['total_memories'] += 1
            self.stats['last_consolidated'] = datetime.now().isoformat()
            
            return memory_id
            
        except Exception as e:
            print(f"Error storing memory: {str(e)}")
            return None

    def retrieve_memories(self, 
                        context: MemoryContext,
                        memory_types: Optional[List[str]] = None,
                        n_results: int = 5) -> List[Dict]:
        """Retrieve memories based on context"""
        try:
            # Default to all memory types if none specified
            if not memory_types:
                memory_types = ['emotional', 'episodic', 'semantic']
                
            memories = []
            
            # Query each specified collection
            for memory_type in memory_types:
                collection = None
                if memory_type == 'emotional':
                    collection = self.emotional_collection
                elif memory_type == 'episodic':
                    collection = self.episodic_collection
                elif memory_type == 'semantic':
                    collection = self.semantic_collection
                
                if collection:
                    # Query with more results to ensure we get enough candidates
                    results = collection.query(
                        query_texts=[context.query],
                        n_results=min(n_results * 3, max(1, self.stats.get(f'{memory_type}_memories', 0))),
                        where={"appropriate": True}
                    )
                    
                    if results and 'documents' in results and len(results['documents']) > 0:
                        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                            # Reconstruct emotional state
                            reconstructed_emotional_state = {
                                'joy': metadata.get('emotional_joy', 0.0),
                                'trust': metadata.get('emotional_trust', 0.0),
                                'fear': metadata.get('emotional_fear', 0.0),
                                'surprise': metadata.get('emotional_surprise', 0.0)
                            }
                            
                            # Reconstruct brain state
                            reconstructed_brain_state = {
                                'arousal': metadata.get('brain_arousal', 0.5),
                                'attention': metadata.get('brain_attention', 0.5),
                                'emotional_valence': metadata.get('brain_emotional_valence', 0.0),
                                'consciousness': metadata.get('brain_consciousness', 1.0),
                                'stress': metadata.get('brain_stress', 0.2),
                                'fatigue': metadata.get('brain_fatigue', 0.0)
                            }
                            
                            # Calculate base semantic similarity
                            semantic_similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
                            
                            # Calculate word overlap boost
                            query_words = set(context.query.lower().split())
                            content_words = set(doc.lower().split())
                            word_overlap = len(query_words.intersection(content_words))
                            word_overlap_boost = 0.1 * word_overlap  # 10% boost per matching word
                            
                            # Calculate emotional similarity
                            emotional_similarity = self._calculate_emotional_similarity(
                                context.emotional_state,
                                reconstructed_emotional_state
                            )
                            
                            # Calculate temporal relevance
                            temporal_relevance = self._calculate_temporal_relevance(
                                datetime.fromisoformat(metadata['timestamp'])
                            )
                            
                            # Calculate developmental relevance
                            developmental_relevance = self._calculate_developmental_relevance(
                                context.age_months,
                                metadata.get('age_months', 0),
                                context.developmental_stage,
                                metadata.get('developmental_stage', '')
                            )
                            
                            # Calculate memory type boost
                            memory_type_boost = 1.0
                            if 'play' in context.query.lower() or 'toys' in context.query.lower():
                                if memory_type == 'emotional':
                                    memory_type_boost = 1.3  # 30% boost for emotional memories about playing
                            elif 'feel' in context.query.lower() or 'emotion' in context.query.lower():
                                if memory_type == 'emotional':
                                    memory_type_boost = 1.2  # 20% boost for emotional memories about feelings
                            
                            # Calculate final relevance score with adjusted weights and boosts
                            base_relevance = (
                                semantic_similarity * 0.4 +  # Base semantic similarity
                                emotional_similarity * 0.3 +  # Emotional context
                                temporal_relevance * 0.2 +  # Recency
                                developmental_relevance * 0.1  # Developmental stage
                            )
                            
                            # Apply boosts
                            relevance_score = base_relevance * (1 + word_overlap_boost) * memory_type_boost
                            
                            memory = {
                                'content': doc,
                                'type': metadata['type'],
                                'metadata': {
                                    'emotional_state': reconstructed_emotional_state,
                                    'brain_state': reconstructed_brain_state,
                                    **{k: v for k, v in metadata.items() 
                                       if k not in ['emotional_joy', 'emotional_trust', 'emotional_fear', 'emotional_surprise',
                                                  'brain_arousal', 'brain_attention', 'brain_emotional_valence',
                                                  'brain_consciousness', 'brain_stress', 'brain_fatigue']}
                                },
                                'relevance': relevance_score,
                                'similarity': semantic_similarity,
                                'emotional_similarity': emotional_similarity,
                                'temporal_relevance': temporal_relevance,
                                'developmental_relevance': developmental_relevance
                            }
                            memories.append(memory)
            
            # Sort memories by relevance score
            memories.sort(key=lambda x: x['relevance'], reverse=True)
            
            return memories[:n_results]
            
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []

    def _calculate_emotional_similarity(self, 
                                     current_state: Dict[str, float],
                                     memory_state: Dict[str, float]) -> float:
        """Calculate similarity between emotional states"""
        if not memory_state:
            return 0.5  # Neutral similarity if no emotional state
            
        # Get vectors of emotion values
        emotions = ['joy', 'trust', 'fear', 'surprise']
        v1 = np.array([current_state.get(e, 0.0) for e in emotions])
        v2 = np.array([memory_state.get(e, 0.0) for e in emotions])
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.5
            
        return float(dot_product / (norm1 * norm2))
        
    def _calculate_temporal_relevance(self, timestamp: datetime) -> float:
        """Calculate temporal relevance of a memory"""
        # More recent memories are more relevant
        time_diff = (datetime.now() - timestamp).total_seconds()
        # Decay over time (days)
        decay = np.exp(-time_diff / (86400 * 7))  # 7 days half-life
        return float(decay)
        
    def _calculate_developmental_relevance(self,
                                        current_age: int,
                                        memory_age: int,
                                        current_stage: str,
                                        memory_stage: str) -> float:
        """Calculate developmental relevance of a memory"""
        # Memories from same stage are more relevant
        stage_relevance = 1.0 if current_stage == memory_stage else 0.5
        
        # Memories from similar age are more relevant
        age_diff = abs(current_age - memory_age)
        age_relevance = np.exp(-age_diff / 6)  # 6 months half-life
        
        return (stage_relevance + age_relevance) / 2

    def clear_memories(self):
        """Clear all stored memories"""
        try:
            # Get all memory IDs from each collection
            emotional_ids = self.emotional_collection.get()['ids'] if self.stats['emotional_memories'] > 0 else []
            episodic_ids = self.episodic_collection.get()['ids'] if self.stats['episodic_memories'] > 0 else []
            semantic_ids = self.semantic_collection.get()['ids'] if self.stats['semantic_memories'] > 0 else []
            
            # Delete memories from each collection
            if emotional_ids:
                self.emotional_collection.delete(ids=emotional_ids)
            if episodic_ids:
                self.episodic_collection.delete(ids=episodic_ids)
            if semantic_ids:
                self.semantic_collection.delete(ids=semantic_ids)
            
            # Reset stats
            self.stats = {
                "total_memories": 0,
                "semantic_memories": 0,
                "episodic_memories": 0,
                "emotional_memories": 0,
                "last_consolidated": None
            }
            
        except Exception as e:
            print(f"Error clearing memories: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return {
            **self.stats,
            'last_updated': datetime.now().isoformat()
        } 