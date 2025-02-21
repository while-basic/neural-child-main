# test_dream_system.py
# Description: Test script for dream simulation system
# Created by: Christopher Celaya

import os
from pathlib import Path
import time
from datetime import datetime, timedelta
from integrated_brain import IntegratedBrain
from dream_system import DreamSystem
from obsidian_api import ObsidianAPI

def print_dream(dream):
    """Print dream content in a formatted way"""
    print("\n" + "="*50)
    print(f"Dream at {dream.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {dream.duration_minutes:.1f} minutes")
    print(f"Type: {dream.__class__.__name__}")
    print("\nNarrative:")
    print(dream.narrative)
    print("\nEmotional Content:")
    print(f"Primary Emotion: {dream.primary_emotion}")
    print(f"Intensity: {dream.emotional_intensity:.2f}")
    print("\nSecondary Emotions:")
    for emotion, score in dream.secondary_emotions.items():
        print(f"- {emotion}: {score:.2f}")
    print("\nSymbols:", ", ".join(dream.dream_symbols))
    if dream.source_memories:
        print("\nSource Memories:", dream.source_memories)
    print("="*50 + "\n")

def main():
    # Set up test environment
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "./test_vault")
    os.makedirs(vault_path, exist_ok=True)
    
    print("Initializing systems...")
    
    # Initialize brain and Obsidian API
    brain = IntegratedBrain()
    obsidian_api = ObsidianAPI(vault_path=vault_path)
    
    # Create test memories in ChromaDB collections
    test_memories = [
        {
            'content': "Today I learned about emotions and how they affect my development.",
            'emotional_intensity': 0.8,
            'id': 'mem_001'
        },
        {
            'content': "I felt very happy when I managed to solve a difficult puzzle.",
            'emotional_intensity': 0.9,
            'id': 'mem_002'
        },
        {
            'content': "I was a bit scared during the thunderstorm but learned about weather.",
            'emotional_intensity': 0.7,
            'id': 'mem_003'
        }
    ]
    
    for memory in test_memories:
        brain.emotional_collection.add(
            documents=[memory['content']],
            metadatas=[{
                'emotional_intensity': memory['emotional_intensity'],
                'id': memory['id'],
                'timestamp': datetime.now().isoformat()
            }],
            ids=[memory['id']]
        )
    
    # Initialize dream system
    dream_system = DreamSystem(brain, obsidian_api)
    
    print("\nGenerating dreams...")
    print("=" * 50)
    
    # Generate different types of dreams
    for i in range(5):
        print(f"\nDream {i+1}/5")
        dream = dream_system.generate_dream()
        if dream:
            print_dream(dream)
        else:
            print("Failed to generate dream")
        
        # Add small delay between dreams
        time.sleep(2)
    
    print("\nTest completed!")
    print(f"Dreams have been recorded in: {vault_path}/Dreams")

if __name__ == "__main__":
    main() 