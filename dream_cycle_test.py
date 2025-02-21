# dream_cycle_test.py
# Description: Test script for running dream cycles in neural child development
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import json
from pathlib import Path
from main import DigitalChild, DevelopmentalStage
from chat_interface import ChatInterface
from emotional_memory_system import EmotionalMemorySystem
from rag_memory import RAGMemorySystem

def get_dream_context(chat: ChatInterface) -> Dict[str, Any]:
    """Get comprehensive context for dream generation"""
    try:
        # Get recent dreams from ChromaDB
        recent_dreams = {'documents': [], 'metadatas': []}
        try:
            recent_dreams = chat.dream_collection.get(
                limit=5,
                include=['documents', 'metadatas']
            )
        except Exception as e:
            print(f"Warning: Could not get recent dreams from ChromaDB: {str(e)}")
        
        # Get recent conversations
        recent_conversations = {'documents': [], 'metadatas': []}
        try:
            recent_conversations = chat.conversation_collection.get(
                limit=5,
                include=['documents', 'metadatas']
            )
        except Exception as e:
            print(f"Warning: Could not get recent conversations from ChromaDB: {str(e)}")
        
        # Get vocabulary context
        try:
            vocab_stats = chat.get_vocabulary_stats()
            known_words = list(chat.known_words)
        except Exception as e:
            print(f"Warning: Could not get vocabulary stats: {str(e)}")
            vocab_stats = {'total_words': 0, 'obsidian_words': 0, 'chroma_words': 0}
            known_words = []
        
        return {
            'recent_dreams': recent_dreams,
            'recent_conversations': recent_conversations,
            'vocab_stats': vocab_stats,
            'known_words': known_words
        }
            
    except Exception as e:
        print(f"Error getting dream context: {str(e)}")
        return {}

def run_developmental_stages(minutes_per_stage: int = 2, cycle_interval_seconds: int = 30):
    """Run dream cycles through all developmental stages"""
    print("Initializing Neural Child Development Stage Progression...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create digital child instance
    child = DigitalChild()
    
    # Get chat interface
    chat = child.chat_interface
    
    # Run through each developmental stage
    for stage in DevelopmentalStage:
        print(f"\n######################################################################")
        print(f"Entering Developmental Stage: {stage.name}")
        print(f"######################################################################\n")
        
        # Set the stage
        child.set_stage(stage)
        
        # Ensure minimum cycle interval
        cycle_interval = max(1, cycle_interval_seconds)  # Minimum 1 second interval
        
        # Calculate number of cycles based on minutes per stage
        cycles = int((minutes_per_stage * 60) / cycle_interval)
        
        # Run dream cycles for this stage
        for cycle in range(cycles):
            # Calculate progress through stage
            progress = cycle / cycles
            
            print(f"\nGenerating Dream #{cycle + 1} (Stage {stage.name} - Progress: {progress:.2f})")
            
            try:
                # Get dream context
                context = get_dream_context(chat)
                
                # Generate dream
                dream = chat._generate_dream()
                print(dream)
                
                # Get vocabulary stats
                vocab_stats = chat.get_vocabulary_stats()
                print("\nVocabulary Stats:")
                print(f"Total Words: {vocab_stats['total_words']}")
                print(f"Obsidian Words: {vocab_stats['obsidian_words']}")
                print(f"ChromaDB Words: {vocab_stats['chroma_words']}\n")
                
                # Get vision development metrics
                vision_metrics = {
                    'visual_acuity': 0.1,
                    'color_perception': 0.0,
                    'depth_perception': 0.0,
                    'pattern_recognition': 0.0,
                    'object_permanence': 0.0,
                    'visual_memory': 0.0,
                    'attention_span': 0.1
                }
                
                print("\nVision Development Metrics:")
                for metric, value in vision_metrics.items():
                    bar = "█" * int(value * 10)
                    print(f"{metric:20}: {bar} {value:.3f}")
                
                # Get emotional state
                print("\nEmotional State:")
                emotions = chat.emotional_context.tolist()
                emotion_names = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Anger', 
                               'Disgust', 'Anticipation', 'Love', 'Guilt', 'Hope', 'Regret']
                for name, value in zip(emotion_names, emotions):
                    bar = "█" * int(value * 10)
                    print(f"{name:12}: {bar} {value:.3f}")
                
                print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Process thoughts about the dream
                print("Processing thoughts about the dream...\n")
                thoughts = [
                    "I had a dream about playing with blocks!",
                    "The colors were so bright and pretty.",
                    "I saw a big friendly face smiling at me.",
                    "There was a warm fuzzy feeling."
                ]
                
                for thought in thoughts:
                    print(f"Thought: {thought}")
                    response = chat.process_message(thought)
                    print(response)
                    
                # Wait for next cycle
                if cycle < cycles - 1:  # Don't wait after last cycle
                    time.sleep(cycle_interval)
                    
            except Exception as e:
                print(f"Error in dream cycle: {str(e)}")
                continue
            
        print(f"\nCompleted {cycles} cycles in {stage.name} stage\n")

if __name__ == "__main__":
    run_developmental_stages(minutes_per_stage=1, cycle_interval_seconds=1)  # Use 1 second interval 