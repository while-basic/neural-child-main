# chat.py
# Description: Command-line interface for interacting with the neural child system
# Created by: Christopher Celaya

import torch
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from main import DigitalChild, MotherLLM, DevelopmentalStage
from conversation_system import ConversationSystem
from logger import DevelopmentLogger
from memory_store import MemoryStore
from ollama_chat import OllamaChildChat
from emotional_regulation import EmotionalState
from obsidian_connector import ObsidianConnector
from emotional_memory_system import EmotionalMemorySystem, EmotionalMemoryEntry, EmotionalAssociation
from brain_state import BrainState
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(child):
    print("\n" + "="*80)
    print(f"Neural Child Development System - Age: {child.age()} months")
    print(f"Current Stage: {child.curriculum.current_stage.name}")
    print(f"Language Stage: {child.language.current_stage.name}")
    print(f"Emotional State: {child.express_feeling()}")
    print("="*80 + "\n")

def print_help():
    print("\nAvailable Commands:")
    print("!help            - Show this help message")
    print("!teach <word/concept> : <definition> - Teach a new word or concept")
    print("!remember <query> - Ask about previously learned memories")
    print("!emotional <memory> - Store an emotional memory")
    print("!dream           - Generate a dream based on recent experiences")
    print("!stats           - Show memory and development statistics")
    print("!stage <stage_name> - Change developmental stage")
    print("!brain           - Show detailed brain state")
    print("!emotions        - Show detailed emotional state")
    print("!vocab           - Show vocabulary statistics")
    print("!save            - Save current state")
    print("!load <filename> - Load saved state")
    print("exit/quit/bye    - Exit the chat")
    print("\nAvailable Stages:")
    for stage in DevelopmentalStage:
        print(f"  {stage.name}")
    print("\n")

def handle_stage_command(child, content):
    try:
        stage_name = content.strip().upper()
        if not stage_name:
            print("\nPlease specify a stage name:")
            for stage in DevelopmentalStage:
                print(f"  {stage.name}")
            return
            
        try:
            new_stage = DevelopmentalStage[stage_name]
            child.set_stage(new_stage, 0.0)
            print(f"\nAdvanced to stage: {new_stage.name}")
            print(f"Age is now: {child.age()} months")
            print(f"Language stage: {child.language.current_stage.name}")
        except KeyError:
            print(f"\nInvalid stage name: {stage_name}")
            print("Available stages:")
            for stage in DevelopmentalStage:
                print(f"  {stage.name}")
    except Exception as e:
        print(f"\nError changing stage: {str(e)}")

def handle_teach_command(memory_store: MemoryStore, model: SentenceTransformer, content: str, child: DigitalChild):
    """Handle the teach command to store semantic memories."""
    try:
        # Parse content
        if ':' not in content:
            print("\nPlease use format: !teach <word/concept> : <definition>")
            return
            
        word, definition = [x.strip() for x in content.split(':', 1)]
        if not word or not definition:
            print("\nBoth word/concept and definition are required.")
            return
        
        # Create embedding
        text = f"{word} - {definition}"
        embedding = model.encode(text, convert_to_tensor=True).cpu().tolist()
        
        # Store semantic memory
        memory_id = memory_store.store_semantic_memory(
            content=text,
            embedding=embedding,
            metadata={
                'word': word,
                'definition': definition,
                'age_months': child.age(),
                'stage': child.curriculum.current_stage.name
            }
        )
        
        if memory_id:
            # Learn the word
            child.language.learn_word(word)
            print(f"\n📚 Learned new word/concept: {word}")
            print(f"📖 Definition: {definition}")
            return {'word': word, 'definition': definition}
        else:
            print("\n❌ Failed to store memory.")
            
    except Exception as e:
        print(f"\n❌ Error teaching word/concept: {str(e)}")

def handle_remember_command(memory_store: MemoryStore, model: SentenceTransformer, query: str):
    """Handle the remember command to retrieve memories."""
    try:
        if not query:
            print("\nPlease provide a query to search memories.")
            return
            
        # Get similar memories
        memories = memory_store.retrieve_similar_memories(query)
        
        if not memories:
            print("\nNo relevant memories found.")
            return
            
        print("\n💭 Found memories:")
        for i, memory in enumerate(memories, 1):
            print(f"\n{i}. {memory['content']}")
            if 'metadata' in memory:
                metadata = memory['metadata']
                if 'type' in metadata:
                    print(f"   Type: {metadata['type'].capitalize()}")
                if 'timestamp' in metadata:
                    print(f"   When: {metadata['timestamp']}")
                if 'stage' in metadata:
                    print(f"   Stage: {metadata['stage']}")
                
    except Exception as e:
        print(f"\n❌ Error retrieving memories: {str(e)}")

def handle_emotional_command(memory_store: MemoryStore, model: SentenceTransformer, content: str, child: DigitalChild):
    """Handle the emotional command to store emotional memories."""
    try:
        if not content:
            print("\nPlease provide content for the emotional memory.")
            return
            
        # Create embedding
        embedding = model.encode(content, convert_to_tensor=True).cpu().tolist()
        
        # Get current emotional state
        emotional_state = {
            'joy': float(child.emotional_state[0]),
            'trust': float(child.emotional_state[1]),
            'fear': float(child.emotional_state[2]),
            'surprise': float(child.emotional_state[3]),
            'sadness': float(child.emotional_state[4]),
            'anger': float(child.emotional_state[5]),
            'disgust': float(child.emotional_state[6]),
            'anticipation': float(child.emotional_state[7])
        }
        
        # Calculate valence and arousal
        valence = (emotional_state['joy'] + emotional_state['trust'] + emotional_state['anticipation'] - 
                  emotional_state['fear'] - emotional_state['sadness'] - emotional_state['disgust']) / 3
        arousal = (emotional_state['surprise'] + emotional_state['fear'] + 
                  emotional_state['anger'] + emotional_state['anticipation']) / 4
        
        # Create emotional memory entry
        memory_entry = EmotionalMemoryEntry(
            content=content,
            emotional_state=emotional_state,
            context=f"Age: {child.age()} months, Stage: {child.curriculum.current_stage.name}",
            intensity=max(emotional_state.values()),
            valence=valence,
            arousal=arousal,
            timestamp=datetime.now(),
            metadata={
                'age_months': child.age(),
                'stage': child.curriculum.current_stage.name,
                'feeling': child.express_feeling(),
                'language_stage': child.language.current_stage.name
            }
        )
        
        # Store in both memory systems for backward compatibility
        memory_id = memory_store.store_emotional_memory(
            content=content,
            embedding=embedding,
            emotional_state=emotional_state,
            metadata=memory_entry.metadata
        )
        
        # Store in emotional memory system
        emotional_memory_id = child.emotional_memory_system.store_memory(memory_entry)
        
        if memory_id and emotional_memory_id:
            print(f"\n💫 Stored emotional memory:")
            print(f"Content: {content}")
            print(f"Feeling: {child.express_feeling()}")
            print("Emotional State:")
            for emotion, value in emotional_state.items():
                bar = "█" * int(value * 10)
                print(f"  {emotion.capitalize():12}: {bar} {value:.2f}")
            print(f"Valence: {'Positive' if valence > 0 else 'Negative'} ({valence:.2f})")
            print(f"Arousal: {'High' if arousal > 0.5 else 'Low'} ({arousal:.2f})")
            
            # Get emotional associations
            associations = child.emotional_memory_system.get_emotional_associations(content)
            if associations:
                print("\nEmotional Associations:")
                for assoc_type, memories in associations.items():
                    if memories:
                        print(f"  {assoc_type}:")
                        for mem in memories[:2]:  # Show top 2 memories
                            print(f"    - {mem['content']}")
            
            return {
                'emotional_state': emotional_state,
                'content': content,
                'feeling': child.express_feeling(),
                'valence': valence,
                'arousal': arousal
            }
        else:
            print("\n❌ Failed to store emotional memory.")
            
    except Exception as e:
        print(f"\n❌ Error storing emotional memory: {str(e)}")

def handle_brain_command(child):
    """Display detailed brain state information"""
    print("\nBrain State Information:")
    
    # Get brain state
    brain_state = child.brain.get_brain_state()
    
    print("\nBrain Activity:")
    metrics = {
        'Arousal': brain_state['arousal'],
        'Attention': brain_state['attention'],
        'Emotional Valence': brain_state['emotional_valence'],
        'Consciousness': brain_state['consciousness'],
        'Stress': brain_state['stress'],
        'Fatigue': brain_state['fatigue']
    }
    
    for metric, value in metrics.items():
        bar = "█" * int(value * 20)
        print(f"  {metric:20}: {bar} {value:.2f}")
    
    print("\nNeurotransmitters:")
    for nt, value in brain_state['neurotransmitters'].items():
        bar = "█" * int(value * 20)
        print(f"  {nt.capitalize():20}: {bar} {value:.2f}")
    
    print("\nDevelopment Metrics:")
    for metric, value in child.metrics.items():
        bar = "█" * int(value * 20)
        print(f"  {metric.replace('_', ' ').title():20}: {bar} {value:.2f}")

def handle_emotions_command(child):
    """Display emotional state information"""
    print("\nEmotional State:")
    emotions = {
        'joy': float(child.emotional_state[0]),
        'trust': float(child.emotional_state[1]),
        'fear': float(child.emotional_state[2]),
        'surprise': float(child.emotional_state[3]),
        'sadness': float(child.emotional_state[4]),
        'anger': float(child.emotional_state[5]),
        'disgust': float(child.emotional_state[6]),
        'anticipation': float(child.emotional_state[7])
    }
    
    # Calculate valence and arousal
    valence = (emotions['joy'] + emotions['trust'] + emotions['anticipation'] - 
              emotions['fear'] - emotions['sadness'] - emotions['disgust']) / 3
    arousal = (emotions['surprise'] + emotions['fear'] + 
              emotions['anger'] + emotions['anticipation']) / 4
    
    # Display emotions with bars
    for emotion, value in emotions.items():
        bar = "█" * int(value * 20)
        print(f"  {emotion.capitalize():12}: {bar} {value:.2f}")
    
    print(f"\nValence: {'Positive' if valence > 0 else 'Negative'} ({valence:.2f})")
    print(f"Arousal: {'High' if arousal > 0.5 else 'Low'} ({arousal:.2f})")
    print(f"Current feeling: {child.express_feeling()}")

def handle_vocab_command(child):
    """Display vocabulary statistics"""
    vocab_stats = child.language.get_vocabulary_stats()
    print("\nVocabulary Statistics:")
    print(f"Total Words Known: {vocab_stats['total_words']}")
    print(f"Active Vocabulary: {vocab_stats['active_words']}")
    print(f"Passive Vocabulary: {vocab_stats['passive_words']}")
    
    if vocab_stats.get('recent_words'):
        print("\nRecently Learned Words:")
        for word, info in vocab_stats['recent_words'].items():
            print(f"  {word}: {info['definition']}")
            print(f"    Confidence: {info['confidence']:.2f}")
            print(f"    Last Used: {info['last_used']}")

def handle_dream_command(child):
    """Generate and display a dream based on recent experiences"""
    try:
        dream = child.brain.dream_generator.generate_dream(
            emotional_state=child.emotional_state,
            recent_experiences=child.memory.get_recent_experiences()
        )
        print("\nDream Generation:")
        print(f"\n💭 {dream['content']}")
        print("\nEmotional Influence:")
        for emotion, value in dream['emotions'].items():
            bar = "█" * int(value * 20)
            print(f"  {emotion.title():12}: {bar} {value:.2f}")
    except Exception as e:
        print(f"\nError generating dream: {str(e)}")

def save_state(child, memory_store, filename=None):
    """Save the current state of the neural child"""
    try:
        if not filename:
            filename = f"digital_child_{child.age()}mo_{int(time.time())}"
        
        save_path = Path("checkpoints") / filename
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state': child.brain.state_dict(),
            'language_state': child.language.state_dict(),
            'curriculum_state': child.curriculum.current_stage,
            'emotional_state': child.emotional_state,
            'timestamp': time.time()
        }, save_path / "model.pth")
        
        # Save memory state
        memory_store.save_state(save_path / "memory_state.json")
        
        print(f"\nState saved successfully to {save_path}")
        return True
    except Exception as e:
        print(f"\nError saving state: {str(e)}")
        return False

def load_state(child, memory_store, filename):
    """Load a previously saved state"""
    try:
        load_path = Path("checkpoints") / filename
        
        if not load_path.exists():
            print(f"\nError: Save file {filename} not found")
            return False
        
        # Load model state
        checkpoint = torch.load(load_path / "model.pth")
        child.brain.load_state_dict(checkpoint['model_state'])
        child.language.load_state_dict(checkpoint['language_state'])
        child.curriculum.current_stage = checkpoint['curriculum_state']
        child.emotional_state = checkpoint['emotional_state']
        
        # Load memory state
        memory_store.load_state(load_path / "memory_state.json")
        
        print(f"\nState loaded successfully from {load_path}")
        return True
    except Exception as e:
        print(f"\nError loading state: {str(e)}")
        return False

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Neural Child Development System')
        parser.add_argument('--stage', type=str, choices=[stage.name for stage in DevelopmentalStage],
                          default='NEWBORN', help='Initial developmental stage')
        parser.add_argument('--progress', type=float, default=0.0,
                          help='Initial progress in the stage (0.0 to 1.0)')
        parser.add_argument('--load', type=str, help='Load saved state from file')
        parser.add_argument('--obsidian-key', type=str, help='Obsidian API key')
        parser.add_argument('--obsidian-vault', type=str, default='neural_child',
                          help='Obsidian vault name')
        parser.add_argument('--obsidian-path', type=str, help='Path to Obsidian vault')
        args = parser.parse_args()
        
        print("Initializing Neural Child Development System...")
        
        # Initialize components
        logger = DevelopmentLogger()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        conversation_system = ConversationSystem(logger)
        memory_store = MemoryStore(persist_directory="memories")
        
        # Create digital child
        initial_stage = DevelopmentalStage[args.stage]
        child = DigitalChild(initial_stage=initial_stage)
        child.set_stage(initial_stage, args.progress)
        
        # Set up conversation systems
        child.mother.conversation_system = conversation_system
        child.conversation_system = conversation_system
        
        # Initialize Ollama chat
        ollama_chat = OllamaChildChat(
            memory_store=memory_store,
            emotional_system=child,
            language_system=child.language,
            model_name="artifish/llama3.2-uncensored"
        )
        
        # Initialize emotional memory system
        print("Initializing emotional memory system...")
        child.emotional_memory_system = EmotionalMemorySystem(
            persist_dir="emotional_memories",
            sentence_transformer_model='all-MiniLM-L6-v2'
        )
        
        # Initialize Obsidian connector if API key is provided
        obsidian = None
        if args.obsidian_key:
            print("Initializing Obsidian connector...")
            # Default to current directory
            default_vault_path = Path.cwd() / "obsidian_vault"
            vault_path = args.obsidian_path if args.obsidian_path else default_vault_path
            
            obsidian = ObsidianConnector(
                logger=logger,
                api_key=args.obsidian_key,
                vault_name=args.obsidian_vault,
                vault_path=vault_path
            )
        
        # Create necessary directories
        os.makedirs("conversations", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("memories", exist_ok=True)
        
        # Load saved state if specified
        if args.load:
            load_state(child, memory_store, args.load)
        
        print("\nSystem initialized successfully!")
        print_header(child)
        print_help()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                print("\n" + "="*80)
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nSaving final state...")
                    save_state(child, memory_store)
                    print("\nGoodbye!")
                    break
                
                if user_input.startswith('!'):
                    if user_input.lower() == '!help':
                        print_help()
                    elif user_input.lower().startswith('!teach '):
                        handle_teach_command(memory_store, model, user_input[7:], child)
                    elif user_input.lower().startswith('!remember '):
                        handle_remember_command(memory_store, model, user_input[10:])
                    elif user_input.lower().startswith('!emotional '):
                        handle_emotional_command(memory_store, model, user_input[11:], child)
                    elif user_input.lower().startswith('!stage '):
                        handle_stage_command(child, user_input[7:])
                    elif user_input.lower() == '!stats':
                        stats = memory_store.get_memory_stats()
                        print("\nMemory Statistics:")
                        print(f"Total Memories: {stats['total_memories']}")
                        print(f"Semantic Memories: {stats['semantic_count']}")
                        print(f"Episodic Memories: {stats['episodic_count']}")
                        print(f"Emotional Memories: {stats['emotional_count']}")
                        print(f"Last Consolidated: {stats['last_consolidated']}")
                    elif user_input.lower() == '!brain':
                        handle_brain_command(child)
                    elif user_input.lower() == '!emotions':
                        handle_emotions_command(child)
                    elif user_input.lower() == '!vocab':
                        handle_vocab_command(child)
                    elif user_input.lower() == '!dream':
                        handle_dream_command(child)
                    elif user_input.lower() == '!save':
                        save_state(child, memory_store)
                    elif user_input.lower().startswith('!load '):
                        load_state(child, memory_store, user_input[6:].strip())
                else:
                    try:
                        print("Processing...", end="\r")
                        response = ollama_chat.chat(user_input)
                        print(" " * 20, end="\r")
                        
                        if not response:
                            print("\nNo response received. Please try again.")
                            continue
                        
                        print("\n💭 Child:", response.get("response", "I'm not sure how to respond."))
                        
                        emotions = response.get("emotions", {})
                        if emotions:
                            print("\n😊 Emotional State:")
                            for emotion, value in emotions.items():
                                bar = "█" * int(value * 10)
                                print(f"  {emotion.capitalize():8}: {bar} {value:.2f}")
                        
                        learning = response.get("learning", {})
                        if learning:
                            new_words = learning.get("new_words", [])
                            new_concepts = learning.get("concepts", [])
                            if new_words:
                                print("\n📚 New Words:", ", ".join(new_words))
                            if new_concepts:
                                print("🧠 New Concepts:", ", ".join(new_concepts))
                        
                        memory = response.get("memory", {})
                        if memory:
                            print(f"\n💫 Formed new {memory['type']} memory:")
                            print(f"  {memory['content']}")
                            bar = "█" * int(memory['emotional_value'] * 10)
                            print(f"  Emotional Value: {bar} {memory['emotional_value']:.2f}")
                        
                        # Store interaction in Obsidian if available
                        if obsidian:
                            obsidian.store_interaction({
                                'user_input': user_input,
                                'response': response.get('response'),
                                'brain_state': response.get('brain_state', {}),
                                'emotions': emotions,
                                'learning': learning,
                                'memory': memory,
                                'stage': child.curriculum.current_stage.name,
                                'age_months': child.age(),
                                'language_stage': child.language.current_stage.name
                            })
                        
                        print("\n" + "="*80)
                    except Exception as chat_error:
                        print(f"\n❌ Error: {str(chat_error)}")
                        print("Please try again.")
                        continue
            except Exception as loop_error:
                print(f"\n❌ Error: {str(loop_error)}")
                print("Continuing with next input...")
                continue
                
    except KeyboardInterrupt:
        print("\nSaving state before exit...")
        try:
            save_state(child, memory_store)
            print("Final state saved. Goodbye!")
        except Exception as save_error:
            print(f"Could not save state: {str(save_error)}")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()