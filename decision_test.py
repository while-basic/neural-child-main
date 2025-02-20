# decision_test.py
# Description: Test script to demonstrate decision network behavior in different scenarios
# Created by: Christopher Celaya

import torch
from main import DigitalChild, DevelopmentalStage
from memory_store import MemoryStore
from logger import DevelopmentLogger
from ollama_chat import OllamaChildChat
from language_development import LanguageDevelopment

def print_decision_output(stage_name: str, response: dict):
    """Print formatted decision output"""
    print(f"\n{'='*80}")
    print(f"Stage: {stage_name}")
    print(f"Response: {response.get('response', 'No response')}")
    
    # Print decision metrics
    decision_metrics = response.get('decision_metrics', {})
    print("\nDecision Metrics:")
    print(f"Confidence: {decision_metrics.get('confidence', 0):.3f}")
    
    action_probs = decision_metrics.get('action_probabilities', [])
    if action_probs:
        print("\nAction Probabilities:")
        for i, prob in enumerate(action_probs):
            print(f"Action {i}: {'█' * int(prob * 20):<20} {prob:.3f}")
    
    # Print emotional state
    emotions = response.get('emotions', {})
    if emotions:
        print("\nEmotional State:")
        for emotion, value in emotions.items():
            print(f"{emotion.capitalize():8}: {'█' * int(value * 20):<20} {value:.3f}")
    
    # Print brain state
    brain_state = response.get('brain_state', {})
    if brain_state:
        print("\nBrain State:")
        print(f"Arousal: {'█' * int(brain_state.get('arousal', 0) * 20):<20} {brain_state.get('arousal', 0):.3f}")
        print(f"Attention: {'█' * int(brain_state.get('attention', 0) * 20):<20} {brain_state.get('attention', 0):.3f}")
        print(f"Emotional Valence: {'█' * int((brain_state.get('emotional_valence', 0) + 1) * 10):<20} {brain_state.get('emotional_valence', 0):.3f}")
    
    print(f"{'='*80}\n")

def test_different_stages():
    """Test decision making across different developmental stages"""
    logger = DevelopmentLogger()
    memory_store = MemoryStore(logger=logger)
    
    stages = [
        DevelopmentalStage.NEWBORN,
        DevelopmentalStage.INFANT,
        DevelopmentalStage.EARLY_TODDLER,
        DevelopmentalStage.LATE_TODDLER,
        DevelopmentalStage.EARLY_PRESCHOOL
    ]
    
    test_messages = [
        "Hello! How are you today?",
        "Do you want to play with blocks?",
        "Look at this bright red ball!",
        "Are you feeling happy?"
    ]
    
    print("\nTesting decisions across developmental stages...")
    
    for stage in stages:
        # Create child at current stage
        child = DigitalChild(initial_stage=stage)
        
        # Initialize chat system
        chat_system = OllamaChildChat(
            memory_store=memory_store,
            emotional_system=child,
            language_system=child.language,
            model_name="artifish/llama3.2-uncensored"
        )
        
        print(f"\nTesting stage: {stage.name}")
        print(f"Age: {child.age()} months")
        print(f"Language stage: {child.language.current_stage.name}")
        
        for message in test_messages:
            print(f"\nUser: {message}")
            response = chat_system.chat(message)
            if response:
                print_decision_output(stage.name, response)
            else:
                print("No response received")

def test_emotional_scenarios():
    """Test decision making with different emotional states"""
    logger = DevelopmentLogger()
    memory_store = MemoryStore(logger=logger)
    
    # Create child at EARLY_TODDLER stage
    child = DigitalChild(initial_stage=DevelopmentalStage.EARLY_TODDLER)
    chat_system = OllamaChildChat(
        memory_store=memory_store,
        emotional_system=child,
        language_system=child.language,
        model_name="artifish/llama3.2-uncensored"
    )
    
    # Test different emotional scenarios
    scenarios = [
        {
            'message': "Would you like a cookie?",
            'emotions': {'joy': 0.8, 'trust': 0.7, 'fear': 0.1, 'surprise': 0.3}
        },
        {
            'message': "There's a loud noise outside!",
            'emotions': {'joy': 0.2, 'trust': 0.3, 'fear': 0.8, 'surprise': 0.7}
        },
        {
            'message': "Let's meet someone new!",
            'emotions': {'joy': 0.5, 'trust': 0.4, 'fear': 0.4, 'surprise': 0.6}
        }
    ]
    
    print("\nTesting decisions with different emotional states...")
    
    for scenario in scenarios:
        # Set emotional state
        child.emotional_state = torch.tensor(
            [scenario['emotions']['joy'],
             scenario['emotions']['trust'],
             scenario['emotions']['fear'],
             scenario['emotions']['surprise']]
        )
        
        print(f"\nScenario with emotional state:")
        for emotion, value in scenario['emotions'].items():
            print(f"{emotion.capitalize():8}: {'█' * int(value * 20):<20} {value:.3f}")
        
        print(f"\nUser: {scenario['message']}")
        response = chat_system.chat(scenario['message'])
        if response:
            print_decision_output("EARLY_TODDLER", response)
        else:
            print("No response received")

def test_memory_influence():
    """Test decision making with different memory contexts"""
    logger = DevelopmentLogger()
    memory_store = MemoryStore(logger=logger)
    
    # Create child at LATE_TODDLER stage
    child = DigitalChild(initial_stage=DevelopmentalStage.LATE_TODDLER)
    chat_system = OllamaChildChat(
        memory_store=memory_store,
        emotional_system=child,
        language_system=child.language,
        model_name="artifish/llama3.2-uncensored"
    )
    
    print("\nTesting decisions with memory influence...")
    
    # First interaction - no memories
    message1 = "Do you like playing with toys?"
    print(f"\nFirst interaction (no memories):")
    print(f"User: {message1}")
    response1 = chat_system.chat(message1)
    if response1:
        print_decision_output("LATE_TODDLER", response1)
    
    # Create some memories
    chat_system._create_memory(
        content="I had fun playing with my red ball",
        memory_type="emotional",
        emotional_value=0.8
    )
    chat_system._create_memory(
        content="I learned that toys are fun to play with",
        memory_type="semantic",
        emotional_value=0.7
    )
    
    # Second interaction - with memories
    print(f"\nSecond interaction (with toy-related memories):")
    print(f"User: {message1}")
    response2 = chat_system.chat(message1)
    if response2:
        print_decision_output("LATE_TODDLER", response2)

def main():
    print("Neural Child Decision Network Test")
    print("="*40)
    
    while True:
        print("\nTest Options:")
        print("1. Test different developmental stages")
        print("2. Test emotional scenarios")
        print("3. Test memory influence")
        print("4. Run all tests")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            test_different_stages()
        elif choice == '2':
            test_emotional_scenarios()
        elif choice == '3':
            test_memory_influence()
        elif choice == '4':
            test_different_stages()
            test_emotional_scenarios()
            test_memory_influence()
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main() 