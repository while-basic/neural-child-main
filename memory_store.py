# memory_store.py
# Description: Vector database for storing and retrieving neural child memories
# Created by: Christopher Celaya

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryStore:
    """A class for storing and retrieving memories using a vector database."""
    
    def __init__(self, persist_directory: str = "memories"):
        """Initialize the memory store with a persistence directory."""
        try:
            self._persist_directory = persist_directory
            os.makedirs(persist_directory, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB with directory: {persist_directory}")
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            
            # Create collections for different types of memories
            self.episodic_memories = self.client.get_or_create_collection(
                name="episodic_memories",
                metadata={"description": "Storage for event-based memories"}
            )
            
            self.semantic_memories = self.client.get_or_create_collection(
                name="semantic_memories",
                metadata={"description": "Storage for factual knowledge"}
            )
            
            self.emotional_memories = self.client.get_or_create_collection(
                name="emotional_memories",
                metadata={"description": "Storage for emotional experiences"}
            )
            
            self.stats = {
                "total_memories": 0,
                "episodic_count": 0,
                "semantic_count": 0,
                "emotional_count": 0,
                "last_consolidated": None
            }
            
            logger.info("Memory store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing memory store: {str(e)}")
            raise
    
    def store_episodic_memory(self, content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Store an episodic memory (event-based experience)"""
        try:
            memory_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['episodic_count']}"
            
            logger.info(f"Storing episodic memory: {content[:50]}...")
            self.episodic_memories.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[{
                    **metadata,
                    "timestamp": datetime.now().isoformat(),
                    "type": "episodic"
                }],
                documents=[content]
            )
            
            self.stats["episodic_count"] += 1
            self.stats["total_memories"] += 1
            logger.info(f"Successfully stored episodic memory with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing episodic memory: {str(e)}")
            raise
    
    def store_semantic_memory(self, content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Store a semantic memory (factual knowledge)"""
        try:
            memory_id = f"sem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['semantic_count']}"
            
            logger.info(f"Storing semantic memory: {content[:50]}...")
            self.semantic_memories.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[{
                    **metadata,
                    "timestamp": datetime.now().isoformat(),
                    "type": "semantic"
                }],
                documents=[content]
            )
            
            self.stats["semantic_count"] += 1
            self.stats["total_memories"] += 1
            logger.info(f"Successfully stored semantic memory with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing semantic memory: {str(e)}")
            raise
    
    def store_emotional_memory(self, content: str, embedding: List[float], 
                             emotional_state: Dict[str, float], metadata: Dict[str, Any]) -> str:
        """Store an emotional memory with associated emotional state"""
        try:
            memory_id = f"em_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['emotional_count']}"
            
            logger.info(f"Storing emotional memory: {content[:50]}...")
            self.emotional_memories.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[{
                    **metadata,
                    **emotional_state,
                    "timestamp": datetime.now().isoformat(),
                    "type": "emotional"
                }],
                documents=[content]
            )
            
            self.stats["emotional_count"] += 1
            self.stats["total_memories"] += 1
            logger.info(f"Successfully stored emotional memory with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing emotional memory: {str(e)}")
            raise
    
    def query_memories(self, query_embedding: List[float], memory_type: str = "all",
                      n_results: int = 5, metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """Query memories based on vector similarity"""
        try:
            results = []
            logger.info(f"Querying memories of type: {memory_type}")
            
            if memory_type in ["all", "episodic"]:
                episodic_results = self.episodic_memories.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, self.stats["episodic_count"] or 1),
                    where=metadata_filter
                )
                results.extend(self._format_results(episodic_results, "episodic"))
            
            if memory_type in ["all", "semantic"]:
                semantic_results = self.semantic_memories.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, self.stats["semantic_count"] or 1),
                    where=metadata_filter
                )
                results.extend(self._format_results(semantic_results, "semantic"))
            
            if memory_type in ["all", "emotional"]:
                emotional_results = self.emotional_memories.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, self.stats["emotional_count"] or 1),
                    where=metadata_filter
                )
                results.extend(self._format_results(emotional_results, "emotional"))
            
            # Sort by distance (similarity)
            results.sort(key=lambda x: x.get("distance", float('inf')))
            logger.info(f"Found {len(results)} matching memories")
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error querying memories: {str(e)}")
            return []
    
    def _format_results(self, results: Dict, memory_type: str) -> List[Dict]:
        """Format query results into a standardized structure"""
        formatted = []
        try:
            if not results or not isinstance(results, dict):
                return formatted
            
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            distances = results.get("distances", [])
            
            if not ids or not documents:
                return formatted
            
            for i in range(len(ids)):
                try:
                    formatted.append({
                        "id": ids[i],
                        "content": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i][0] if i < len(distances) and distances[i] else float('inf'),
                        "type": memory_type
                    })
                except Exception as e:
                    logger.error(f"Error formatting result {i}: {str(e)}")
                    continue
                    
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}")
            return formatted
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate memories and update importance based on access patterns"""
        # Get current timestamp
        current_time = datetime.now()
        
        # Update consolidation timestamp
        self.stats["last_consolidated"] = current_time.isoformat()
        
        # Return consolidation statistics
        return {
            "timestamp": current_time.isoformat(),
            "total_memories": self.stats["total_memories"],
            "episodic_count": self.stats["episodic_count"],
            "semantic_count": self.stats["semantic_count"],
            "emotional_count": self.stats["emotional_count"]
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return self.stats
    
    def save_state(self, file_path: str):
        """Save the current state of the memory store to a JSON file."""
        state = {
            "statistics": self.stats,
            "persist_directory": self._persist_directory
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, file_path: str):
        """Load the memory store state from a JSON file."""
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        self.stats = state["statistics"]
        self._persist_directory = state["persist_directory"]
        
        # Reinitialize the client with the loaded persist directory
        self.client = chromadb.Client(Settings(persist_directory=self._persist_directory))
        self.collection = self.client.get_or_create_collection(name="memories")
        
        # Create collections for different types of memories
        self.episodic_memories = self.client.get_or_create_collection(
            name="episodic_memories",
            metadata={"description": "Storage for event-based memories"}
        )
        
        self.semantic_memories = self.client.get_or_create_collection(
            name="semantic_memories",
            metadata={"description": "Storage for factual knowledge"}
        )
        
        self.emotional_memories = self.client.get_or_create_collection(
            name="emotional_memories",
            metadata={"description": "Storage for emotional experiences"}
        ) 