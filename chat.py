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

def clear_screen():
    pass

def print_header(child):
    print("\n" + "="*80)
    print(f"Neural Child Development System - Age: {child.age()} months")
    print(f"Current Stage: {child.curriculum.current_stage.name}")
    
    # Get brain state
    brain_state = child.brain.get_brain_state()
    
    # Display brain state summary
    print("\nBrain State:")
    print(f"  Arousal: {'█' * int(brain_state['arousal'] * 10):<10} {brain_state['arousal']:.2f}")
    print(f"  Attention: {'█' * int(brain_state['attention'] * 10):<10} {brain_state['attention']:.2f}")
    print(f"  Emotional Valence: {'█' * int((brain_state['emotional_valence'] + 1) * 5):<10} {brain_state['emotional_valence']:.2f}")
    print(f"  Consciousness: {'█' * int(brain_state['consciousness'] * 10):<10} {brain_state['consciousness']:.2f}")
    
    # Display neurotransmitter levels
    print("\nNeurotransmitters:")
    for nt, level in brain_state['neurotransmitters'].items():
        print(f"  {nt.capitalize()}: {'█' * int(level * 10):<10} {level:.2f}")
    
    # Display emotional state
    print("\nEmotional State:")
    print(f"  {child.express_feeling()}")
    
    print("="*80 + "\n")

def print_help():
    print("\nAvailable Commands:")
    print("!help - Show this help message")
    print("!teach <word/concept> : <definition> - Teach a new word or concept")
    print("!remember <query> - Ask about previously learned memories")
    print("!emotional <memory> - Store an emotional memory")
    print("!stats - Show memory statistics")
    print("!brain - Show detailed brain state")
    print("!stage <stage_name> - Change developmental stage")
    print("!network - Generate neural network visualization")
    print("exit/quit/bye - Exit the chat")
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
            'surprise': float(child.emotional_state[3])
        }
        
        # Calculate valence and arousal
        valence = (emotional_state['joy'] + emotional_state['trust'] - emotional_state['fear']) / 2
        arousal = (emotional_state['surprise'] + emotional_state['fear']) / 2
        
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
                print(f"  {emotion.capitalize():8}: {bar} {value:.2f}")
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

def handle_brain_command(child: DigitalChild):
    """Handle the brain command to show detailed brain state"""
    stats = child.get_brain_development_stats()
    
    print("\nBrain Development Statistics:")
    print("\nBrain State:")
    for key, value in stats['brain_state'].items():
        if isinstance(value, dict):
            print(f"\n{key.capitalize()}:")
            for k, v in value.items():
                if isinstance(v, float):
                    bar = "█" * int(v * 10)
                    print(f"  {k.capitalize():15}: {bar} {v:.2f}")
        elif isinstance(value, float):
            bar = "█" * int(value * 10)
            print(f"  {key.capitalize():15}: {bar} {value:.2f}")
            
    print("\nDevelopment Metrics:")
    for key, value in stats['metrics'].items():
        bar = "█" * int(value * 10)
        print(f"  {key.replace('_', ' ').capitalize():15}: {bar} {value:.2f}")
        
    print("\nIntegrated Metrics:")
    for key, value in stats['integrated_metrics'].items():
        bar = "█" * int(value * 10)
        print(f"  {key.replace('_', ' ').capitalize():15}: {bar} {value:.2f}")
        
    print(f"\nDevelopment Stage: {stats['development_stage']}")
    print(f"Age: {stats['age_months']} months")
    print(f"Language Stage: {stats['language_stage']}")
    
    print("\nEmotional Memory Statistics:")
    for key, value in stats['emotional_stats'].items():
        if isinstance(value, int):
            print(f"  {key.replace('_', ' ').capitalize():15}: {value}")

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Neural Child Development System')
        parser.add_argument('--stage', type=str, choices=[stage.name for stage in DevelopmentalStage],
                          default='NEWBORN', help='Initial developmental stage')
        parser.add_argument('--progress', type=float, default=0.0,
                          help='Initial progress in the stage (0.0 to 1.0)')
        parser.add_argument('--obsidian-key', type=str, help='Obsidian API key')
        parser.add_argument('--obsidian-vault', type=str, default='neural_child',
                          help='Obsidian vault name')
        parser.add_argument('--obsidian-path', type=str, help='Path to Obsidian vault')
        args = parser.parse_args()
        
        print("Initializing logger...")
        logger = DevelopmentLogger()
        
        print("Initializing sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Initializing conversation system...")
        conversation_system = ConversationSystem(logger)
        
        print("Initializing Neural Child Development System...")
        initial_stage = DevelopmentalStage[args.stage]
        child = DigitalChild(initial_stage=initial_stage)
        child.set_stage(initial_stage, args.progress)
        child.mother.conversation_system = conversation_system
        child.conversation_system = conversation_system
        
        print("Initializing memory store...")
        memory_store = MemoryStore(logger=logger)
        
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
        
        print("Initializing Ollama chat system...")
        ollama_chat = OllamaChildChat(
            memory_store=memory_store,
            emotional_system=child,
            language_system=child.language,
            model_name="artifish/llama3.2-uncensored"
        )
        
        print("Creating necessary directories...")
        os.makedirs("conversations", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("memories", exist_ok=True)
        
        print("\nSystem initialized successfully!")
        print_header(child)
        print_help()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                print("\n" + "="*80)
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nSaving final state...")
                    final_conversation_path = f"conversations/final_conversation_{child.age()}mo.json"
                    conversation_system.save_conversation(final_conversation_path)
                    memory_store.save_state("memories/final_memory_state.json")
                    
                    # Get emotional memory stats
                    emotional_stats = child.emotional_memory_system.get_memory_stats()
                    print("\nEmotional Memory Statistics:")
                    print(f"Total Memories: {emotional_stats['total_memories']}")
                    print(f"Positive Memories: {emotional_stats['positive_memories']}")
                    print(f"Negative Memories: {emotional_stats['negative_memories']}")
                    print(f"Complex Memories: {emotional_stats['complex_memories']}")
                    print(f"Traumatic Memories: {emotional_stats['traumatic_memories']}")
                    
                    save_path = f"checkpoints/digital_child_{child.age()}mo_final"
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({
                        'model_state': child.brain.state_dict(),
                        'language_state': child.language.state_dict(),
                        'curriculum_state': child.curriculum.current_stage,
                        'emotional_state': child.emotional_state,
                        'emotional_memory_stats': emotional_stats,
                        'timestamp': time.time()
                    }, f"{save_path}/model.pth")
                    
                    print("Goodbye!")
                    break
                
                if user_input.startswith('!'):
                    if user_input.lower() == '!help':
                        print_help()
                    elif user_input.lower().startswith('!teach '):
                        result = handle_teach_command(memory_store, model, user_input[7:], child)
                        if obsidian and result:
                            obsidian.store_language_learning({
                                'description': f"Learned new word: {result['word']} - {result['definition']}",
                                'category': 'vocabulary',
                                'word': result['word'],
                                'definition': result['definition'],
                                'stage': child.curriculum.current_stage.name,
                                'age_months': child.age(),
                                'language_stage': child.language.current_stage.name
                            })
                    elif user_input.lower().startswith('!remember '):
                        handle_remember_command(memory_store, model, user_input[10:])
                    elif user_input.lower().startswith('!emotional '):
                        result = handle_emotional_command(memory_store, model, user_input[11:], child)
                        if obsidian and result:
                            obsidian.store_emotional_state({
                                'description': result['content'],
                                'primary_emotion': result['feeling'],
                                'emotional_state': result['emotional_state'],
                                'stage': child.curriculum.current_stage.name,
                                'age_months': child.age(),
                                'language_stage': child.language.current_stage.name
                            })
                    elif user_input.lower().startswith('!stage '):
                        old_stage = child.curriculum.current_stage.name
                        handle_stage_command(child, user_input[7:])
                        if obsidian and child.curriculum.current_stage.name != old_stage:
                            obsidian.store_development({
                                'description': f"Advanced to stage {child.curriculum.current_stage.name}",
                                'stage': child.curriculum.current_stage.name,
                                'previous_stage': old_stage,
                                'age_months': child.age(),
                                'language_stage': child.language.current_stage.name,
                                'metrics': child.metrics
                            })
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
                    elif user_input.lower() == '!network':
                        print("\nGenerating neural network visualization...")
                        if obsidian:
                            child.export_network_visualization(obsidian.vault_path)
                            print("✨ Network visualization created in Obsidian vault")
                            print("Open the Network folder in Obsidian to view the visualization")
                        else:
                            print("❌ Obsidian integration not enabled. Please provide an API key.")
                    else:
                        try:
                            print("Processing...", end="\r")
                            response = ollama_chat.chat(user_input)
                            print(" " * 20, end="\r")
                            
                            if not response:
                                print("\nNo response received. Please try again.")
                                continue
                            
                            print("\n💭 Child:", response.get("response", "I'm not sure how to respond."))
                            
                            # Display brain state changes
                            brain_state = response.get("brain_state", {})
                            print("\n🧠 Brain State Changes:")
                            print(f"  Arousal: {'█' * int(brain_state.get('arousal', 0) * 10):<10} {brain_state.get('arousal', 0):.2f}")
                            print(f"  Attention: {'█' * int(brain_state.get('attention', 0) * 10):<10} {brain_state.get('attention', 0):.2f}")
                            print(f"  Emotional Valence: {'█' * int((brain_state.get('emotional_valence', 0) + 1) * 5):<10} {brain_state.get('emotional_valence', 0):.2f}")
                            
                            # Display emotional state
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
                                    'brain_state': brain_state,
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
                else:
                    try:
                        print("Processing...", end="\r")
                        response = ollama_chat.chat(user_input)
                        print(" " * 20, end="\r")
                        
                        if not response:
                            print("\nNo response received. Please try again.")
                            continue
                        
                        print("\n💭 Child:", response.get("response", "I'm not sure how to respond."))
                        
                        # Display brain state changes
                        brain_state = response.get("brain_state", {})
                        print("\n🧠 Brain State Changes:")
                        print(f"  Arousal: {'█' * int(brain_state.get('arousal', 0) * 10):<10} {brain_state.get('arousal', 0):.2f}")
                        print(f"  Attention: {'█' * int(brain_state.get('attention', 0) * 10):<10} {brain_state.get('attention', 0):.2f}")
                        print(f"  Emotional Valence: {'█' * int((brain_state.get('emotional_valence', 0) + 1) * 5):<10} {brain_state.get('emotional_valence', 0):.2f}")
                        
                        # Display emotional state
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
                                'brain_state': brain_state,
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
            save_path = f"checkpoints/digital_child_{child.age()}mo_final"
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'model_state': child.brain.state_dict(),
                'language_state': child.language.state_dict(),
                'curriculum_state': child.curriculum.current_stage,
                'emotional_state': child.emotional_state,
                'timestamp': time.time()
            }, f"{save_path}/model.pth")
            memory_store.save_state("memories/final_memory_state.json")
            print("Final state saved. Goodbye!")
        except Exception as save_error:
            print(f"Could not save state: {str(save_error)}")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()