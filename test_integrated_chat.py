# test_integrated_chat.py
# Description: Test script for the integrated chat system
# Created by: Christopher Celaya

import torch
import sys
from integrated_chat import IntegratedChatSystem
from developmental_stages import DevelopmentalStage

def main():
    print("\n=== Neural Child Development System ===")
    print("Initializing integrated chat system...")
    
    # Initialize chat system
    chat = IntegratedChatSystem(initial_stage=DevelopmentalStage.EARLY_CHILDHOOD)
    
    print("\nNeural Child is ready for interaction!")
    
    # Process piped input if any, but don't exit
    if not sys.stdin.isatty():
        message = sys.stdin.read().strip()
        if message:
            try:
                response = chat.process_message(message)
                print(response)
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    print("\nCommands:")
    print("  'exit' - End conversation")
    print("  'save' - Save current brain state")
    print("  'load <filename>' - Load brain state from file")
    print("  'memories <query>' - Search emotional memories")
    print("  'state' - Show current brain state")
    print("  'history' - Show conversation history")
    print("\n" + "="*40 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for commands
            if user_input.lower() == 'exit':
                print("\nGoodbye! Saving final brain state...")
                chat.save_state()
                break
                
            elif user_input.lower() == 'save':
                chat.save_state()
                print("Brain state saved!")
                continue
                
            elif user_input.lower().startswith('load '):
                filename = user_input[5:].strip()
                chat.load_state(filename)
                print(f"Brain state loaded from {filename}")
                continue
                
            elif user_input.lower().startswith('memories '):
                query = user_input[9:].strip()
                memories = chat.get_emotional_memories(query)
                print("\nRelevant emotional memories:")
                for i, memory in enumerate(memories, 1):
                    print(f"\n{i}. Content: {memory.content}")
                    print(f"   Feeling: {memory.emotional_state}")
                    print(f"   When: {memory.timestamp}")
                continue
                
            elif user_input.lower() == 'state':
                brain_state = chat.get_brain_state()
                print("\nCurrent Brain State:")
                for key, value in brain_state.items():
                    print(f"{key}: {value}")
                continue
                
            elif user_input.lower() == 'history':
                history = chat.get_conversation_history()
                print("\nConversation History:")
                for entry in history[-5:]:  # Show last 5 exchanges
                    print(f"\n{entry['role']}: {entry['content']}")
                    print(f"Time: {entry['timestamp']}")
                continue
            
            # Process regular message
            response = chat.process_message(user_input)
            print(response)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")
            continue

if __name__ == "__main__":
    main() 