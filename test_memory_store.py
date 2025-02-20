# test_memory_store.py
# Description: Test script for the vector-based memory system
# Created by: Christopher Celaya

import torch
from memory_store import MemoryStore
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json

def test_memory_storage():
    """Test basic memory storage and retrieval"""
    print("\nTesting Memory Storage System...")
    
    # Initialize memory store
    memory_store = MemoryStore(persist_directory="test_memories")
    
    # Initialize sentence transformer for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test data
    test_memories = {
        "episodic": [
            "Mommy played peek-a-boo with me today",
            "I saw a colorful butterfly in the garden",
            "Daddy read me a bedtime story about dragons"
        ],
        "semantic": [
            "Dogs say woof and cats say meow",
            "The sky is blue because of light scattering",
            "Circles are round and have no corners"
        ],
        "emotional": [
            "I felt happy when mommy hugged me",
            "I was scared of the loud thunder",
            "I was excited to play with my new toy"
        ]
    }
    
    # Test emotional states
    emotional_states = [
        {"joy": 0.8, "trust": 0.7, "fear": 0.1, "surprise": 0.3},
        {"joy": 0.2, "trust": 0.3, "fear": 0.8, "surprise": 0.6},
        {"joy": 0.9, "trust": 0.8, "fear": 0.1, "surprise": 0.7}
    ]
    
    print("\nStoring test memories...")
    memory_ids = []
    
    # Store episodic memories
    for memory in test_memories["episodic"]:
        embedding = model.encode(memory).tolist()
        memory_id = memory_store.store_episodic_memory(
            content=memory,
            embedding=embedding,
            metadata={"category": "daily_experience"}
        )
        memory_ids.append(memory_id)
        print(f"Stored episodic memory: {memory}")
    
    # Store semantic memories
    for memory in test_memories["semantic"]:
        embedding = model.encode(memory).tolist()
        memory_id = memory_store.store_semantic_memory(
            content=memory,
            embedding=embedding,
            metadata={"category": "knowledge"}
        )
        memory_ids.append(memory_id)
        print(f"Stored semantic memory: {memory}")
    
    # Store emotional memories
    for memory, emotion in zip(test_memories["emotional"], emotional_states):
        embedding = model.encode(memory).tolist()
        memory_id = memory_store.store_emotional_memory(
            content=memory,
            embedding=embedding,
            emotional_state=emotion,
            metadata={"category": "emotional_experience"}
        )
        memory_ids.append(memory_id)
        print(f"Stored emotional memory: {memory}")
    
    print("\nTesting memory retrieval...")
    
    # Test queries
    test_queries = [
        "playing with mommy",
        "learning about animals",
        "feeling scared"
    ]
    
    for query in test_queries:
        print(f"\nQuerying memories similar to: {query}")
        query_embedding = model.encode(query).tolist()
        
        results = memory_store.query_memories(
            query_embedding=query_embedding,
            n_results=3
        )
        
        for result in results:
            print(f"Found memory: {result['content']}")
            print(f"Type: {result['type']}")
            print(f"Distance: {result['distance']:.4f}")
            print("---")
    
    # Test memory consolidation
    print("\nTesting memory consolidation...")
    consolidation_stats = memory_store.consolidate_memories()
    print("Consolidation stats:", json.dumps(consolidation_stats, indent=2))
    
    # Test memory statistics
    print("\nMemory statistics:")
    stats = memory_store.get_memory_stats()
    print(json.dumps(stats, indent=2))
    
    # Save and load state
    print("\nTesting state saving and loading...")
    memory_store.save_state("test_memories/memory_state.json")
    
    new_memory_store = MemoryStore(persist_directory="test_memories")
    new_memory_store.load_state("test_memories/memory_state.json")
    
    print("Loaded memory stats:", json.dumps(new_memory_store.get_memory_stats(), indent=2))

def main():
    """Run all memory system tests"""
    print("Starting Memory System Tests...")
    
    # Create test directories
    os.makedirs("test_memories", exist_ok=True)
    
    # Run tests
    test_memory_storage()
    
    print("\nAll memory system tests completed!")

if __name__ == "__main__":
    main() 