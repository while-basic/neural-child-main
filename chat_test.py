# chat_test.py
# Description: Test script for interacting with the neural child chat system
# Created by: Christopher Celaya

import torch
from main import DigitalChild, DevelopmentalStage
from memory_store import MemoryStore
from logger import DevelopmentLogger
from ollama_chat import OllamaChildChat

def print_state(child, chat_system):
    """Print the current state of the child"""
    print("\n" + "="*80)
    print(f"Neural Child Development System - Age: {child.age()} months")
    print(f"Stage: {child.curriculum.current_stage.name}")
    print(f"Language Stage: {child.language.current_stage.name}")
    
    # Get brain state
    brain_state = child.brain.brain_state
    
    # Display brain state summary
    print("\nBrain State:")
    print(f"  Arousal: {'█' * int(brain_state.arousal * 10):<10} {brain_state.arousal:.2f}")
    print(f"  Attention: {'█' * int(brain_state.attention * 10):<10} {brain_state.attention:.2f}")
    print(f"  Emotional Valence: {'█' * int((brain_state.emotional_valence + 1) * 5):<10} {brain_state.emotional_valence:.2f}")
    print(f"  Consciousness: {'█' * int(brain_state.consciousness * 10):<10} {brain_state.consciousness:.2f}")
    
    # Display emotional state
    emotions = child.emotional_state.cpu().tolist()
    print("\nEmotional State:")
    print(f"  Joy: {'█' * int(emotions[0] * 10):<10} {emotions[0]:.2f}")
    print(f"  Trust: {'█' * int(emotions[1] * 10):<10} {emotions[1]:.2f}")
    print(f"  Fear: {'█' * int(emotions[2] * 10):<10} {emotions[2]:.2f}")
    print(f"  Surprise: {'█' * int(emotions[3] * 10):<10} {emotions[3]:.2f}")
    
    # Display known names
    if chat_system.known_names:
        print("\nKnown People:")
        for name, relation in chat_system.known_names.items():
            print(f"  - {name}: {relation}")
    
    print("="*80 + "\n")

def main():
    # Initialize components
    print("Initializing neural child system...")
    logger = DevelopmentLogger()
    memory_store = MemoryStore(logger=logger)
    
    # Create child at NEWBORN stage
    child = DigitalChild(initial_stage=DevelopmentalStage.NEWBORN)
    
    # Initialize chat system
    chat_system = OllamaChildChat(
        memory_store=memory_store,
        emotional_system=child,
        language_system=child.language,
        model_name="artifish/llama3.2-uncensored"  # Using llama2 model
    )
    
    print("\nNeural child system initialized!")
    print("\nYou can now chat with Alex. Type 'exit' to end the conversation.")
    print("\nSpecial commands:")
    print("  !teach <word/concept> : <definition> - Teach a new word or concept")
    print("  !remember <query> - Ask about memories")
    print("  !state - Show current state")
    print("  !help - Show this help message")
    
    # Initial state display
    print_state(child, chat_system)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
            
            # Check for special commands
            if user_input.startswith('!'):
                if user_input == '!state':
                    print_state(child, chat_system)
                    continue
                elif user_input == '!help':
                    print("\nSpecial commands:")
                    print("  !teach <word/concept> : <definition> - Teach a new word or concept")
                    print("  !remember <query> - Ask about memories")
                    print("  !state - Show current state")
                    print("  !help - Show this help message")
                    continue
            
            # Process chat message
            print("\nProcessing...", end="\r")
            response = chat_system.chat(user_input)
            print(" " * 20, end="\r")  # Clear processing message
            
            if response:
                # Display response
                print(f"\nAlex: {response['response']}")
                
                # Display emotional changes
                if 'emotions' in response:
                    print("\nEmotional Response:")
                    for emotion, value in response['emotions'].items():
                        bar = "█" * int(value * 10)
                        print(f"  {emotion.capitalize():8}: {bar} {value:.2f}")
                
                # Display learning
                if 'learning' in response:
                    learning = response['learning']
                    if learning.get('new_words'):
                        print("\nLearned new words:", ", ".join(learning['new_words']))
                    if learning.get('concepts'):
                        print("Learned new concepts:", ", ".join(learning['concepts']))
                
                # Display memory formation
                if 'memory' in response and response['memory']:
                    memory = response['memory']
                    print(f"\nFormed new {memory['type']} memory:")
                    print(f"  Content: {memory['content']}")
                    print(f"  Emotional value: {'█' * int(memory['emotional_value'] * 10)} {memory['emotional_value']:.2f}")
                
                # Display brain state changes
                brain_state = response.get('brain_state', {})
                print("\nBrain State Changes:")
                print(f"  Arousal: {'█' * int(brain_state.get('arousal', 0) * 10):<10} {brain_state.get('arousal', 0):.2f}")
                print(f"  Attention: {'█' * int(brain_state.get('attention', 0) * 10):<10} {brain_state.get('attention', 0):.2f}")
                print(f"  Emotional Valence: {'█' * int((brain_state.get('emotional_valence', 0) + 1) * 5):<10} {brain_state.get('emotional_valence', 0):.2f}")
                
                # Display level up if occurred
                if 'level_up' in response:
                    level_up = response['level_up']
                    print("\n" + "="*80)
                    print(level_up['message'])
                    print("\nAchievements:")
                    for achievement in level_up['achievements']:
                        print(f"  {achievement}")
                    print("="*80)
            else:
                print("\nAlex seems unable to respond right now.")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Continuing...")
            continue

if __name__ == "__main__":
    main() 