# test_emotional_chat.py
# Description: Test script for emotional chat system with heartbeat responses
# Created by: Christopher Celaya

import os
import time
from digital_child import DigitalChild
from obsidian_api import ObsidianAPI
from emotional_chat_system import EmotionalChatSystem

def print_response(response):
    """Pretty print the chat response"""
    print("\n=== Response ===")
    print(f"Heartbeat: {response['heartbeat'].get('current_rate', 'N/A')} BPM")
    print(f"State: {response['heartbeat'].get('state', 'N/A')}")
    print(f"Emotional Impact: {response['heartbeat'].get('emotional_impact', 'N/A')}")
    print("\nBrain State:")
    brain_state = response['brain_state']
    print(f"- Emotional Valence: {brain_state.get('emotional_valence', 'N/A')}")
    print(f"- Arousal: {brain_state.get('arousal', 'N/A')}")
    print(f"- Attention: {brain_state.get('attention', 'N/A')}")
    print("=" * 20)

def main():
    # Set up Obsidian vault path
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "./test_vault")
    os.makedirs(vault_path, exist_ok=True)
    
    print("Initializing systems...")
    obsidian_api = ObsidianAPI(vault_path=vault_path)
    child = DigitalChild()
    chat_system = EmotionalChatSystem(child.brain, obsidian_api)
    
    # Test different emotional scenarios
    test_messages = [
        # Happy/Positive messages
        "Hello! I'm so happy to meet you!",
        "You're doing great! I love how you're learning.",
        
        # Negative/Angry messages
        "I'm very angry right now!",
        "This is terrible, I hate it!",
        
        # Fearful/Anxious messages
        "I'm scared of what might happen...",
        "I'm feeling very anxious about this.",
        
        # Surprise messages
        "Wow! That's amazing!",
        "OMG! I can't believe it!",
        
        # Memory commands
        "!remember Today I learned about emotions and how they affect my heartbeat",
        "!reflect",
        
        # Mixed emotions
        "I'm excited but also a bit nervous",
        "This is wonderful but it makes me anxious"
    ]
    
    print("\nStarting emotional chat test...")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}/{len(test_messages)}")
        print(f"Message: {message}")
        
        # Process message and get response
        response = chat_system.process_message(message)
        print_response(response)
        
        # Add small delay between messages
        time.sleep(2)
    
    print("\nTest completed!")
    print(f"Memories have been recorded in: {vault_path}")

if __name__ == "__main__":
    main() 