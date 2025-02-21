# emotional_memory_system.py
# Description: Advanced emotional memory system with ChromaDB integration for neural child development
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer

@dataclass
class EmotionalMemoryEntry:
    """Represents a single emotional memory entry"""
    content: str
    emotional_state: Dict[str, float]
    context: str
    intensity: float
    valence: float  # -1 to 1, negative to positive
    arousal: float  # 0 to 1, low to high activation
    timestamp: datetime
    metadata: Dict[str, Any]

class EmotionalAssociation(Enum):
    """Types of emotional associations that can be formed"""
    POSITIVE = auto()
    NEGATIVE = auto()
    NEUTRAL = auto()
    COMPLEX = auto()
    TRAUMATIC = auto()

class EmotionalMemoryProcessor(nn.Module):
    """Neural network for processing emotional memories"""
    def __init__(self, input_dim: int = 384, emotion_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.emotion_dim = emotion_dim
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Emotional embedding network
        self.emotion_encoder = nn.Sequential(
            nn.Linear(input_dim + emotion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, emotion_dim)
        )
        
        # Memory consolidation network
        self.consolidation_net = nn.Sequential(
            nn.Linear(emotion_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, emotion_dim)
        )
        
        # Association strength predictor
        self.association_net = nn.Sequential(
            nn.Linear(emotion_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, text_embedding: torch.Tensor, emotional_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Ensure dimensions are correct
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.unsqueeze(0)  # Add batch dimension
        if len(emotional_state.shape) == 1:
            emotional_state = emotional_state.unsqueeze(0)  # Add batch dimension
            
        # Combine text and emotional features
        combined = torch.cat([text_embedding, emotional_state], dim=-1)
        
        # Generate emotional embedding
        emotional_embedding = self.emotion_encoder(combined)
        
        # Process memory consolidation
        consolidated_memory = self.consolidation_net(
            torch.cat([emotional_embedding, emotional_state], dim=-1)
        )
        
        # Calculate association strength
        association_strength = self.association_net(
            torch.cat([emotional_embedding, consolidated_memory], dim=-1)
        )
        
        return {
            'emotional_embedding': emotional_embedding,
            'consolidated_memory': consolidated_memory,
            'association_strength': association_strength
        }

class EmotionalMemorySystem:
    """Advanced emotional memory system using ChromaDB"""
    def __init__(self, 
                chat_interface,
                sentence_transformer_model: str = "all-MiniLM-L6-v2"):
        """Initialize the emotional memory system"""
        # Store brain reference
        self.brain = chat_interface
        
        # Use existing ChromaDB client from chat interface
        self.chroma_client = chat_interface.chroma_client
        
        # Use existing collections
        self.emotional_collection = chat_interface.emotional_memory_collection
        self.episodic_collection = chat_interface.episodic_memory_collection
        self.semantic_collection = chat_interface.semantic_memory_collection
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model)
        
        # Memory statistics
        self._load_memory_stats()
        
    def _load_memory_stats(self):
        """Load and update memory statistics from collections"""
        self.stats = {
            'total_memories': 0,
            'emotional_memories': 0,
            'positive_memories': 0,
            'negative_memories': 0,
            'neutral_memories': 0,
            'complex_memories': 0,
            'traumatic_memories': 0,
            'last_consolidated': None
        }
        
        # Update stats from emotional memories
        try:
            emotional_memories = self.emotional_collection.get()
            if emotional_memories and 'metadatas' in emotional_memories:
                self.stats['total_memories'] = len(emotional_memories['metadatas'])
                self.stats['emotional_memories'] = len(emotional_memories['metadatas'])
                for metadata in emotional_memories['metadatas']:
                    if metadata:
                        # Calculate emotional association
                        joy = metadata.get('emotional_joy', 0.0)
                        trust = metadata.get('emotional_trust', 0.0)
                        fear = metadata.get('emotional_fear', 0.0)
                        surprise = metadata.get('emotional_surprise', 0.0)
                        intensity = metadata.get('intensity', 0.0)
                        
                        # Determine association type
                        if fear > 0.7 and intensity > 0.8:
                            self.stats['traumatic_memories'] += 1
                        elif (joy + trust) / 2 > 0.5 and fear < 0.3:
                            self.stats['positive_memories'] += 1
                        elif fear > 0.5:
                            self.stats['negative_memories'] += 1
                        elif abs((joy + trust) / 2 - fear) < 0.2:
                            self.stats['neutral_memories'] += 1
                        else:
                            self.stats['complex_memories'] += 1
                            
        except Exception as e:
            print(f"Warning: Could not load emotional memory stats: {str(e)}")
            
    def store_memory(self, memory: EmotionalMemoryEntry) -> str:
        """Store an emotional memory with its associated states"""
        try:
            # Generate memory ID
            memory_id = f"em_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['total_memories']}"
            
            # Get embedding
            embedding = self.sentence_transformer.encode(memory.content)
            
            # Flatten emotional and brain states into metadata
            flattened_metadata = {
                'type': 'emotional',  # Store type directly in metadata
                'timestamp': memory.timestamp.isoformat(),
                'timestamp_unix': memory.timestamp.timestamp(),  # Add Unix timestamp for comparison
                'appropriate': True,
                
                # Emotional state
                'emotional_joy': float(memory.emotional_state.get('joy', 0.0)),
                'emotional_trust': float(memory.emotional_state.get('trust', 0.0)),
                'emotional_fear': float(memory.emotional_state.get('fear', 0.0)),
                'emotional_surprise': float(memory.emotional_state.get('surprise', 0.0)),
                
                # Memory properties
                'intensity': float(memory.intensity),
                'valence': float(memory.valence),
                'arousal': float(memory.arousal),
                'context': memory.context
            }
            
            # Add additional metadata
            if memory.metadata:
                for key, value in memory.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        flattened_metadata[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (str, int, float, bool)):
                                flattened_metadata[f"{key}_{subkey}"] = subvalue
            
            # Store in emotional collection
            self.emotional_collection.add(
                embeddings=[embedding.tolist()],
                documents=[memory.content],
                metadatas=[flattened_metadata],
                ids=[memory_id]
            )
            
            # Update statistics
            self.stats['total_memories'] += 1
            self.stats['emotional_memories'] += 1
            
            # Calculate emotional association
            association = self._calculate_emotional_association(memory)
            if association == EmotionalAssociation.POSITIVE:
                self.stats['positive_memories'] += 1
            elif association == EmotionalAssociation.NEGATIVE:
                self.stats['negative_memories'] += 1
            elif association == EmotionalAssociation.NEUTRAL:
                self.stats['neutral_memories'] += 1
            elif association == EmotionalAssociation.COMPLEX:
                self.stats['complex_memories'] += 1
            elif association == EmotionalAssociation.TRAUMATIC:
                self.stats['traumatic_memories'] += 1
            
            # Update last consolidated timestamp
            self.stats['last_consolidated'] = datetime.now().isoformat()
            
            return memory_id
            
        except Exception as e:
            print(f"Error storing memory: {str(e)}")
            return None
            
    def retrieve_similar_memories(self, 
                                query: str, 
                                emotional_state: Optional[Dict[str, float]] = None,
                                n_results: int = 5) -> List[Dict]:
        """Retrieve memories similar to the query, optionally filtered by emotional state"""
        try:
            # Get total count of memories
            total_memories = self.stats.get('total_memories', 0)
            if total_memories == 0:
                return []
                
            # Query emotional memories first
            try:
                emotional_results = self.emotional_collection.query(
                    query_texts=[query],
                    n_results=min(n_results * 2, max(1, total_memories)),
                    where={"appropriate": True} if total_memories > 0 else None
                )
            except Exception as e:
                print(f"Warning: Error querying emotional memories: {str(e)}")
                emotional_results = None
            
            # Process results
            memories = []
            
            # Process emotional memories
            if emotional_results and 'documents' in emotional_results and len(emotional_results['documents']) > 0:
                for i, (doc, metadata) in enumerate(zip(emotional_results['documents'][0], emotional_results['metadatas'][0])):
                    if not self._is_content_appropriate(doc):
                        continue
                        
                    reconstructed_emotional_state = {
                        'joy': metadata.get('emotional_joy', 0.0),
                        'trust': metadata.get('emotional_trust', 0.0),
                        'fear': metadata.get('emotional_fear', 0.0),
                        'surprise': metadata.get('emotional_surprise', 0.0)
                    }
                    
                    # Calculate emotional similarity if emotional_state provided
                    emotional_similarity = 1.0
                    if emotional_state:
                        emotional_similarity = self._calculate_emotional_similarity(
                            emotional_state,
                            reconstructed_emotional_state
                        )
                    
                    memory = {
                        'content': doc,
                        'metadata': {
                            **metadata,
                            'emotional_state': reconstructed_emotional_state,
                            'type': 'emotional'
                        },
                        'similarity': emotional_results['distances'][0][i] if 'distances' in emotional_results else 0.5,
                        'emotional_similarity': emotional_similarity
                    }
                    memories.append(memory)
            
            # Sort memories by combined similarity
            if memories:
                memories.sort(key=lambda x: (
                    x.get('emotional_similarity', 0.5) * 0.7 +  # Weight emotional similarity more
                    (1 - (x.get('similarity', 0.5) or 0.5)) * 0.3  # ChromaDB distances are 0=similar, 1=different
                ), reverse=True)
            
            return memories[:n_results]
            
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []
            
    def get_emotional_associations(self, query: str) -> Dict[str, List[Dict]]:
        """Get emotional associations for a given query"""
        try:
            # Initialize empty associations
            associations = {
                EmotionalAssociation.POSITIVE.name: [],
                EmotionalAssociation.NEGATIVE.name: [],
                EmotionalAssociation.NEUTRAL.name: [],
                EmotionalAssociation.COMPLEX.name: [],
                EmotionalAssociation.TRAUMATIC.name: []
            }
            
            # Get total count of memories
            total_memories = self.stats['total_memories']
            if total_memories == 0:
                return associations
                
            # Query memories with adjusted n_results and filter for appropriate content
            results = self.emotional_collection.query(
                query_texts=[query],
                n_results=min(10, total_memories),  # Adjust n_results based on available memories
                where={"appropriate": True}  # Only get appropriate memories
            )
            
            # Process results
            if results and 'documents' in results and len(results['documents']) > 0:
                seen_contents = set()  # Track unique contents
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    # Skip if we've seen this content before
                    if doc in seen_contents:
                        continue
                    seen_contents.add(doc)
                    
                    # Double-check content appropriateness
                    if not self._is_content_appropriate(doc):
                        continue
                        
                    association = metadata.get('association', 'NEUTRAL')
                    if association in associations:
                        # Reconstruct emotional state
                        emotional_state = {
                            'joy': metadata.get('emotional_joy', 0.0),
                            'trust': metadata.get('emotional_trust', 0.0),
                            'fear': metadata.get('emotional_fear', 0.0),
                            'surprise': metadata.get('emotional_surprise', 0.0)
                        }
                        
                        associations[association].append({
                            'content': doc,
                            'metadata': {
                                **metadata,
                                'emotional_state': emotional_state
                            }
                        })
            
            return associations
            
        except Exception as e:
            print(f"Error getting emotional associations: {str(e)}")
            return associations
            
    def get_memory_timeline(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get memories within a specific time range"""
        try:
            # Query all memories
            results = self.emotional_collection.get()
            
            # Filter by date range
            timeline = []
            for doc, metadata in zip(results['documents'], results['metadatas']):
                memory_date = datetime.fromisoformat(metadata['timestamp'])
                if start_date <= memory_date <= end_date:
                    timeline.append({
                        'content': doc,
                        'metadata': metadata,
                        'timestamp': memory_date
                    })
            
            # Sort by timestamp
            timeline.sort(key=lambda x: x['timestamp'])
            return timeline
            
        except Exception as e:
            print(f"Error getting memory timeline: {str(e)}")
            return []
    
    def _calculate_emotional_association(self, memory: EmotionalMemoryEntry) -> EmotionalAssociation:
        """Calculate the emotional association type for a memory.
        
        Args:
            memory (EmotionalMemoryEntry): The memory to analyze
            
        Returns:
            EmotionalAssociation: The type of emotional association
        """
        # Extract emotional values
        joy = memory.emotional_state.get('joy', 0.0)
        trust = memory.emotional_state.get('trust', 0.0)
        fear = memory.emotional_state.get('fear', 0.0)
        surprise = memory.emotional_state.get('surprise', 0.0)
        
        # Calculate emotional metrics
        positive_affect = (joy + trust) / 2
        negative_affect = fear
        emotional_complexity = abs(positive_affect - negative_affect)
        
        # Determine association type
        if fear > 0.7 and memory.intensity > 0.8:
            return EmotionalAssociation.TRAUMATIC
        elif positive_affect > 0.5 and fear < 0.3:
            return EmotionalAssociation.POSITIVE
        elif fear > 0.5:
            return EmotionalAssociation.NEGATIVE
        elif emotional_complexity < 0.2:
            return EmotionalAssociation.NEUTRAL
        else:
            return EmotionalAssociation.COMPLEX
            
    def _analyze_emotional_patterns(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional patterns from heartbeat history.
        
        Args:
            history (List[Dict]): List of heartbeat history entries
            
        Returns:
            Dict containing pattern analysis
        """
        if not history:
            return {'status': 'No history available'}
            
        # Calculate statistics
        rates = [entry['rate'] for entry in history]
        states = [entry['state'] for entry in history]
        
        # Calculate emotional stability
        rate_stability = 1.0 - (np.std(rates) / np.mean(rates)) if rates else 0.0
        
        # Calculate state transitions
        transitions = len([i for i in range(1, len(states)) if states[i] != states[i-1]])
        transition_rate = transitions / len(states) if states else 0.0
        
        # Identify dominant states
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1
            
        dominant_state = max(state_counts.items(), key=lambda x: x[1])[0] if state_counts else None
        
        # Calculate emotional resilience (ability to return to baseline)
        baseline_returns = len([i for i in range(1, len(states)) 
                              if states[i] == 'RESTING' and states[i-1] != 'RESTING'])
        resilience = baseline_returns / transitions if transitions > 0 else 0.0
        
        analysis = {
            'average_rate': np.mean(rates) if rates else 0.0,
            'rate_variance': np.var(rates) if rates else 0.0,
            'rate_stability': rate_stability,
            'dominant_state': dominant_state,
            'state_transitions': transitions,
            'transition_rate': transition_rate,
            'emotional_resilience': resilience,
            'period_analyzed': f"{len(history)} entries",
            'state_distribution': state_counts
        }
        
        return analysis
        
    def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate recent memories and update emotional patterns.
        
        Returns:
            Dict containing consolidation results
        """
        try:
            # Get recent memories
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            recent_memories = self.emotional_collection.get(
                where={"timestamp_unix": {"$gt": cutoff_time}}
            )
            
            if not recent_memories or 'documents' not in recent_memories:
                return {'status': 'No recent memories to consolidate'}
                
            # Group memories by emotional association
            memory_groups = {
                'positive': [],
                'negative': [],
                'neutral': [],
                'complex': [],
                'traumatic': []
            }
            
            for doc, metadata in zip(recent_memories['documents'], recent_memories['metadatas']):
                # Reconstruct emotional state
                emotional_state = {
                    'joy': metadata.get('emotional_joy', 0.0),
                    'trust': metadata.get('emotional_trust', 0.0),
                    'fear': metadata.get('emotional_fear', 0.0),
                    'surprise': metadata.get('emotional_surprise', 0.0)
                }
                
                # Create memory entry
                memory = EmotionalMemoryEntry(
                    content=doc,
                    emotional_state=emotional_state,
                    context=metadata.get('context', ''),
                    intensity=metadata.get('intensity', 0.0),
                    valence=metadata.get('valence', 0.0),
                    arousal=metadata.get('arousal', 0.0),
                    timestamp=datetime.fromtimestamp(metadata.get('timestamp_unix', datetime.now().timestamp())),
                    metadata=metadata
                )
                
                # Determine association and group
                association = self._calculate_emotional_association(memory)
                memory_groups[association.name.lower()].append(memory)
                
            # Calculate consolidation metrics
            total_memories = len(recent_memories['documents'])
            consolidation_metrics = {
                'total_memories': total_memories,
                'positive_ratio': len(memory_groups['positive']) / total_memories if total_memories > 0 else 0.0,
                'negative_ratio': len(memory_groups['negative']) / total_memories if total_memories > 0 else 0.0,
                'emotional_complexity': len(memory_groups['complex']) / total_memories if total_memories > 0 else 0.0,
                'trauma_exposure': len(memory_groups['traumatic']) / total_memories if total_memories > 0 else 0.0
            }
            
            # Update brain's emotional state based on consolidated memories
            self._update_emotional_state_from_consolidation(memory_groups, consolidation_metrics)
            
            return {
                'status': 'Memory consolidation complete',
                'metrics': consolidation_metrics,
                'memory_distribution': {k: len(v) for k, v in memory_groups.items()}
            }
            
        except Exception as e:
            print(f"Error during memory consolidation: {str(e)}")
            return {'status': 'Error during consolidation', 'error': str(e)}
            
    def _update_emotional_state_from_consolidation(self, 
                                                 memory_groups: Dict[str, List[EmotionalMemoryEntry]],
                                                 metrics: Dict[str, float]):
        """Update brain's emotional state based on consolidated memories.
        
        Args:
            memory_groups (Dict[str, List[EmotionalMemoryEntry]]): Grouped memories
            metrics (Dict[str, float]): Consolidation metrics
        """
        try:
            # Calculate base emotional adjustments
            joy_adjustment = metrics['positive_ratio'] - metrics['negative_ratio']
            trust_adjustment = max(0, 0.5 - metrics['trauma_exposure']) * 0.5
            fear_adjustment = metrics['trauma_exposure'] * 0.3
            surprise_adjustment = metrics['emotional_complexity'] * 0.2
            
            # Get current heartbeat info
            heartbeat_info = self.brain.heartbeat.get_current_heartbeat()
            current_rate = heartbeat_info.get('rate', 80)
            
            # Calculate heartbeat adjustments
            if current_rate > 100:  # Elevated heart rate
                joy_adjustment *= 0.8  # Reduce positive emotions
                fear_adjustment *= 1.2  # Increase fear
            elif current_rate < 70:  # Low heart rate
                trust_adjustment *= 1.2  # Increase trust
                fear_adjustment *= 0.8  # Reduce fear
                
            # Update brain's emotional state with smoothing
            self.brain.brain_state.emotional_valence = max(-1.0, min(1.0,
                self.brain.brain_state.emotional_valence * 0.7 +  # 70% previous state
                (joy_adjustment - fear_adjustment) * 0.3  # 30% new adjustment
            ))
            
            self.brain.brain_state.arousal = max(0.0, min(1.0,
                self.brain.brain_state.arousal * 0.7 +
                (surprise_adjustment + fear_adjustment) * 0.3
            ))
            
            # Update neurotransmitter levels
            self.brain.brain_state.neurotransmitters['dopamine'] = max(0.1, min(1.0,
                self.brain.brain_state.neurotransmitters['dopamine'] * 0.8 +
                joy_adjustment * 0.2
            ))
            
            self.brain.brain_state.neurotransmitters['serotonin'] = max(0.1, min(1.0,
                self.brain.brain_state.neurotransmitters['serotonin'] * 0.8 +
                trust_adjustment * 0.2
            ))
            
            self.brain.brain_state.neurotransmitters['norepinephrine'] = max(0.1, min(1.0,
                self.brain.brain_state.neurotransmitters['norepinephrine'] * 0.8 +
                (fear_adjustment + surprise_adjustment) * 0.2
            ))
            
        except Exception as e:
            print(f"Error updating emotional state from consolidation: {str(e)}")
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        # Ensure stats are loaded
        if not hasattr(self, 'stats'):
            self._load_memory_stats()
            
        return {
            **self.stats,
            'last_updated': datetime.now().isoformat()
        }
        
    def clear_memories(self):
        """Clear all stored memories"""
        self.emotional_collection.delete(ids=self.emotional_collection.get()['ids'])
        self.stats = {k: 0 for k in self.stats}
        
    def _is_emotion_appropriate(self, emotional_state: Dict[str, float]) -> bool:
        """Check if emotional intensity is appropriate for development"""
        # Calculate overall emotional intensity
        intensity = sum(emotional_state.values()) / len(emotional_state)
        
        # Check for extreme emotions
        max_emotion = max(emotional_state.values())
        min_emotion = min(emotional_state.values())
        emotional_range = max_emotion - min_emotion
        
        # Thresholds for emotional appropriateness
        MAX_INTENSITY = 0.8
        MAX_RANGE = 0.6
        
        return intensity <= MAX_INTENSITY and emotional_range <= MAX_RANGE
        
    def _is_content_appropriate(self, content: str) -> bool:
        """Check if content is appropriate for child development"""
        # List of inappropriate words and patterns
        inappropriate_patterns = [
            # Profanity and explicit content
            r'\b(fuck|shit|damn|bitch|cunt|ass|dick|cock|pussy|sex|porn)\b',
            # Violence
            r'\b(kill|murder|die|death|dead|blood|gore)\b',
            # Substances
            r'\b(drug|cocaine|heroin|weed|marijuana)\b',
            # Abuse
            r'\b(rape|molest|abuse)\b',
            # Additional inappropriate concepts
            r'\b(hate|evil|cruel|torture)\b'
        ]
        
        # Convert to lowercase for case-insensitive matching
        content = content.lower()
        
        # Check each pattern
        for pattern in inappropriate_patterns:
            if re.search(pattern, content):
                return False
                
        return True 

    def process_emotional_input(self, input_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process emotional input and store in memory"""
        try:
            # Get current emotional state
            emotional_state = {
                'joy': float(self.emotional_state[0]),
                'trust': float(self.emotional_state[1]),
                'fear': float(self.emotional_state[2]),
                'surprise': float(self.emotional_state[3])
            }
            
            # Calculate valence and arousal
            valence = (emotional_state['joy'] + emotional_state['trust'] - emotional_state['fear']) / 2
            arousal = (emotional_state['surprise'] + emotional_state['fear']) / 2
            
            # Calculate intensity based on emotional extremes
            intensity = max(
                emotional_state['fear'],
                emotional_state['surprise'],
                abs(valence)
            )
            
            # Create emotional memory entry
            memory_entry = EmotionalMemoryEntry(
                content=input_text,
                emotional_state=emotional_state,
                context=context or f"Age: {self.age()} months, Stage: {self.curriculum.current_stage.name}",
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                timestamp=datetime.now(),
                metadata={
                    'age_months': self.age(),
                    'stage': self.curriculum.current_stage.name,
                    'feeling': self.express_feeling(),
                    'language_stage': self.language.current_stage.name
                }
            )
            
            # Store memory
            memory_id = self.store_memory(memory_entry)
            
            # Get emotional associations
            associations = self.get_emotional_associations(input_text)
            
            # Update emotional state based on associations
            self._update_emotional_state_from_associations(associations)
            
            return {
                'memory_id': memory_id,
                'emotional_state': emotional_state,
                'valence': valence,
                'arousal': arousal,
                'intensity': intensity,
                'associations': associations
            }
            
        except Exception as e:
            print(f"Error processing emotional input: {str(e)}")
            return {} 

    def _calculate_emotional_similarity(self, state1: Dict[str, float], state2: Dict[str, float]) -> float:
        """Calculate similarity between two emotional states"""
        # Get vectors of emotion values
        v1 = np.array([state1.get(k, 0.0) for k in ['joy', 'trust', 'fear', 'surprise']])
        v2 = np.array([state2.get(k, 0.0) for k in ['joy', 'trust', 'fear', 'surprise']])
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2)) 

    def _update_emotional_state_from_associations(self, associations: Dict[str, List[Dict]]):
        """Update emotional state based on emotional associations"""
        try:
            # Calculate influence weights for each association type
            weights = {
                EmotionalAssociation.POSITIVE.name: 0.3,  # Increased positive influence
                EmotionalAssociation.NEGATIVE.name: -0.15,  # Reduced negative influence
                EmotionalAssociation.NEUTRAL.name: 0.0,
                EmotionalAssociation.COMPLEX.name: 0.1,
                EmotionalAssociation.TRAUMATIC.name: -0.2  # Reduced traumatic influence
            }
            
            # Calculate total influence
            total_influence = torch.zeros(4, device=self.device)
            total_weight = 0
            
            # Track memory age influence
            now = datetime.now()
            
            for assoc_type, memories in associations.items():
                if memories and assoc_type in weights:
                    weight = weights[assoc_type]
                    for memory in memories:
                        # Calculate memory age factor (newer memories have more influence)
                        memory_time = datetime.fromisoformat(memory['metadata'].get('timestamp', now.isoformat()))
                        time_diff = (now - memory_time).total_seconds()
                        age_factor = max(0.1, min(1.0, 1.0 / (1.0 + time_diff / 3600)))  # Decay over hours
                        
                        emotional_state = memory['metadata']['emotional_state']
                        influence = torch.tensor([
                            emotional_state['joy'],
                            emotional_state['trust'],
                            emotional_state['fear'],
                            emotional_state['surprise']
                        ], device=self.device)
                        
                        # Apply age-weighted influence
                        total_influence += influence * weight * age_factor
                        total_weight += abs(weight) * age_factor
            
            if total_weight > 0:
                # Apply influence with adaptive smoothing
                current_intensity = self.emotional_state.mean().item()
                smoothing = max(0.2, min(0.5, 1.0 - current_intensity))  # More smoothing when calm
                
                self.emotional_state = (1 - smoothing) * self.emotional_state + smoothing * (
                    self.emotional_state + total_influence / total_weight
                )
                
                # Ensure values stay in valid range with minimum values
                self.emotional_state = torch.clamp(self.emotional_state, 0.1, 1.0)
                
                # Reduce fear over time unless reinforced
                if 'fear' in self.unique_emotions:
                    fear_idx = 2  # Index of fear in emotional state
                    self.emotional_state[fear_idx] = max(
                        0.2,  # Minimum fear
                        min(0.8,  # Maximum fear
                            self.emotional_state[fear_idx] * 0.95  # Gradual reduction
                        )
                    )
                
        except Exception as e:
            print(f"Error updating emotional state from associations: {str(e)}")
            
    def _calculate_emotional_association(self, memory: EmotionalMemoryEntry) -> EmotionalAssociation:
        """Calculate the emotional association type for a memory"""
        # Extract emotional values
        joy = memory.emotional_state.get('joy', 0)
        trust = memory.emotional_state.get('trust', 0)
        fear = memory.emotional_state.get('fear', 0)
        surprise = memory.emotional_state.get('surprise', 0)
        
        # Calculate emotional metrics
        positive_emotions = (joy + trust) / 2
        negative_emotions = (fear + max(0, surprise - 0.5)) / 2
        emotional_intensity = max(fear, surprise, memory.intensity)
        
        # Determine association type with adjusted thresholds
        if fear > 0.7 and emotional_intensity > 0.8:  # Higher threshold for traumatic
            return EmotionalAssociation.TRAUMATIC
        elif positive_emotions > 0.5 and negative_emotions < 0.3:  # Lower threshold for positive
            return EmotionalAssociation.POSITIVE
        elif negative_emotions > 0.5:  # Adjusted negative threshold
            return EmotionalAssociation.NEGATIVE
        elif abs(positive_emotions - negative_emotions) < 0.2:
            return EmotionalAssociation.NEUTRAL
        else:
            return EmotionalAssociation.COMPLEX 