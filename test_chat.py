# test_chat.py
# Description: Test script for the neural child development system chat interface
# Created by: Christopher Celaya

import torch
from main import DigitalChild, DevelopmentalStage
from chat_interface import ChatInterface
from datetime import datetime

def print_emotional_state(emotional_context):
    """Pretty print the emotional state"""
    emotions = [
        "Joy", "Trust", "Fear", "Surprise", 
        "Sadness", "Anger", "Disgust", "Anticipation",
        "Love", "Guilt", "Hope", "Regret"
    ]
    print("\nEmotional State:")
    for emotion, value in zip(emotions, emotional_context.tolist()):
        print(f"{emotion:12}: {value:.3f}")
    print()

def print_vocabulary_stats(chat):
    """Print vocabulary statistics"""
    stats = chat.get_vocabulary_stats()
    print("\nVocabulary Stats:")
    print(f"Total Words: {stats['total_words']}")
    print(f"Obsidian Words: {stats['obsidian_words']}")
    print(f"ChromaDB Words: {stats['chroma_words']}")
    print()

def main():
    print("Initializing Neural Child Development System...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create digital child instance
    child = DigitalChild()
    
    # Set initial development stage
    child.set_stage(DevelopmentalStage.EARLY_CHILDHOOD, progress=0.5)
    
    # Create chat interface
    chat = ChatInterface(child.brain, device=device)
    
    print("\nNeural Child is ready for interaction!")
    print("Type 'exit' to end the conversation")
    print("Type 'dream' to generate a dream")
    print("Type 'emotions' to see current emotional state")
    print("Type 'vocab' to see vocabulary statistics")
    print("Type 'time' to see time awareness information")
    print("\n" + "="*50 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for special commands
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'dream':
                dream = chat._generate_dream()
                print(f"\nChild: {dream}")
            elif user_input.lower() == 'emotions':
                print_emotional_state(chat.emotional_context)
            elif user_input.lower() == 'vocab':
                print_vocabulary_stats(chat)
            elif user_input.lower() == 'time':
                current_time = datetime.now()
                time_since_last = current_time - chat.last_interaction_time
                time_since_creation = current_time - chat.creation_time
                
                print("\nTime Awareness:")
                print(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Time Since Last Interaction: {time_since_last}")
                print(f"Time Since Creation: {time_since_creation}")
                if chat.favorite_interaction_times:
                    print("\nFavorite Interaction Times:")
                    for hour, count in chat.favorite_interaction_times:
                        print(f"  {hour:02d}:00 - {count} interactions")
                print()
            else:
                # Process regular message
                response = chat.process_message(user_input)
                print(f"\nChild: {response}")
                
                # Show emotional state after response
                print_emotional_state(chat.emotional_context)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

if __name__ == "__main__":
    main() 