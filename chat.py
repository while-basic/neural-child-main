# chat.py
# Description: Command-line interface for interacting with the neural child system
# Created by: Christopher Celaya

import torch
import os
import time
import argparse
from datetime import datetime
from sentence_transformers import SentenceTransformer
from main import DigitalChild, MotherLLM, DevelopmentalStage
from conversation_system import ConversationSystem
from logger import DevelopmentLogger
from memory_store import MemoryStore
from ollama_chat import OllamaChildChat
from emotional_regulation import EmotionalState

def clear_screen():
    pass

def print_header(child):
    print("\n" + "="*80)
    print(f"Neural Child Development System - Age: {child.age()} months")
    print(f"Current Stage: {child.curriculum.current_stage.name}")
    print(f"Emotional State: {child.express_feeling()}")
    print("="*80 + "\n")

def print_help():
    print("\nAvailable Commands:")
    print("!help - Show this help message")
    print("!teach <word/concept> : <definition> - Teach a new word or concept")
    print("!remember <query> - Ask about previously learned memories")
    print("!emotional <memory> - Store an emotional memory")
    print("!stats - Show memory statistics")
    print("!stage <stage_name> - Change developmental stage")
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

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Neural Child Development System')
        parser.add_argument('--stage', type=str, choices=[stage.name for stage in DevelopmentalStage],
                          default='NEWBORN', help='Initial developmental stage')
        parser.add_argument('--progress', type=float, default=0.0,
                          help='Initial progress in the stage (0.0 to 1.0)')
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
        memory_store = MemoryStore(persist_directory="memories")
        
        print("Initializing Ollama chat system...")
        ollama_chat = OllamaChildChat(
            memory_store=memory_store,
            emotional_system=child,
            language_system=child.language,
            model_name="llama3"
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
                    
                    save_path = f"checkpoints/digital_child_{child.age()}mo_final"
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({
                        'model_state': child.brain.state_dict(),
                        'language_state': child.language.state_dict(),
                        'curriculum_state': child.curriculum.current_stage,
                        'emotional_state': child.emotional_state,
                        'timestamp': time.time()
                    }, f"{save_path}/model.pth")
                    
                    print("Goodbye!")
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