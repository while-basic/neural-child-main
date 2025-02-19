# test_language_development.py
# Description: Test script for language development system
# Created by: Christopher Celaya

import torch
import numpy as np
from language_development import LanguageDevelopment, LanguageStage, WordCategory
import json
import os
from datetime import datetime

def test_language_progression():
    """Test language stage progression and development"""
    print("\nTesting Language Development System...")
    
    # Initialize language system
    language = LanguageDevelopment(device='cuda')
    
    # Test data directory
    os.makedirs("test_results", exist_ok=True)
    
    # Sample emotional states
    emotional_states = {
        'happy': torch.tensor([0.8, 0.7, 0.1, 0.3]),  # High joy, high trust
        'scared': torch.tensor([0.3, 0.4, 0.8, 0.6]),  # High fear
        'curious': torch.tensor([0.5, 0.6, 0.2, 0.8]),  # High surprise
        'neutral': torch.tensor([0.5, 0.5, 0.5, 0.5])   # Neutral
    }
    
    # Test progression through stages
    test_results = {
        'stages': [],
        'vocabulary_growth': [],
        'utterances': [],
        'metrics': []
    }
    
    # Test mother's utterances for different stages
    mother_utterances = [
        # Prelinguistic stage utterances
        "Oh, my sweet baby! Are you hungry?",
        "Look at you, such a happy baby!",
        "Time for a nap, my little one.",
        
        # Holophrastic stage utterances
        "Do you want more milk?",
        "The ball is red. Can you say ball?",
        "Daddy is coming home soon!",
        
        # Telegraphic stage utterances
        "Let's build a tower with blocks.",
        "The cat is sleeping on the bed.",
        "Would you like to play with your toys?",
        
        # Multiword stage utterances
        "Yesterday we went to the park and saw ducks.",
        "When you're happy, you make mommy very happy too.",
        "Let's read your favorite story about the little bear."
    ]
    
    print("\nTesting language progression through stages...")
    for stage in ['NEWBORN', 'INFANT', 'TODDLER', 'EARLY_CHILDHOOD']:
        print(f"\nTesting {stage} stage:")
        
        # Process multiple utterances in this stage
        for utterance in mother_utterances:
            # Randomly select emotional state
            emotion = np.random.choice(list(emotional_states.keys()))
            emotional_state = emotional_states[emotion]
            
            # Learn from interaction
            result = language.learn_from_interaction(utterance, emotional_state, stage)
            
            # Generate response
            response = language.generate_utterance(utterance, emotional_state)
            
            print(f"Mother: {utterance}")
            print(f"Baby ({emotion}): {response}")
            print(f"Stage Progress: {result['stage_progress']:.2f}")
            
            # Record results
            test_results['stages'].append({
                'developmental_stage': stage,
                'language_stage': result['current_stage'],
                'progress': result['stage_progress']
            })
            
            test_results['vocabulary_growth'].append({
                'stage': stage,
                'vocabulary_size': result['metrics']['vocabulary_size']
            })
            
            test_results['utterances'].append({
                'stage': stage,
                'mother_utterance': utterance,
                'baby_response': response,
                'emotional_state': emotion
            })
            
            test_results['metrics'].append(result['metrics'])
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"test_results/language_development_{timestamp}.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest results saved to test_results/language_development_{timestamp}.json")
    
    # Print summary
    print("\nLanguage Development Test Summary:")
    print(f"Final Vocabulary Size: {test_results['vocabulary_growth'][-1]['vocabulary_size']}")
    print(f"Final Language Stage: {test_results['stages'][-1]['language_stage']}")
    print("\nSample utterances from different stages:")
    for utterance in test_results['utterances'][::len(mother_utterances)]:
        print(f"{utterance['stage']}: {utterance['baby_response']}")

def test_word_learning():
    """Test word learning and categorization"""
    print("\nTesting Word Learning System...")
    
    language = LanguageDevelopment(device='cuda')
    
    # Test words for each category
    test_words = {
        WordCategory.NOUN: ['ball', 'dog', 'book', 'toy'],
        WordCategory.VERB: ['run', 'jump', 'play', 'eat'],
        WordCategory.ADJECTIVE: ['big', 'red', 'happy', 'soft'],
        WordCategory.PRONOUN: ['me', 'you', 'it', 'we'],
        WordCategory.PREPOSITION: ['in', 'on', 'at', 'by'],
        WordCategory.ARTICLE: ['a', 'the'],
        WordCategory.ADVERB: ['quickly', 'softly', 'happily'],
        WordCategory.INTERJECTION: ['oh', 'wow', 'oops', 'yay']
    }
    
    results = {
        'categorization_accuracy': {},
        'learned_vocabulary': {}
    }
    
    print("\nTesting word categorization:")
    for category, words in test_words.items():
        correct = 0
        for word in words:
            categorized = language._categorize_word(word)
            if categorized == category:
                correct += 1
            print(f"Word: {word:10} Expected: {category.name:12} Got: {categorized.name if categorized else 'None':12}")
        
        accuracy = correct / len(words)
        results['categorization_accuracy'][category.name] = accuracy
        print(f"\n{category.name} Accuracy: {accuracy:.2%}")
    
    # Test learning
    print("\nTesting word learning:")
    emotional_state = torch.tensor([0.6, 0.7, 0.2, 0.4])  # Positive emotional state
    
    for category, words in test_words.items():
        learned_words = []
        for word in words:
            # Create a simple sentence using the word
            sentence = f"This is a {word}."
            result = language.learn_from_interaction(sentence, emotional_state, 'TODDLER')
            if word in result['new_words']:
                learned_words.append(word)
        
        results['learned_vocabulary'][category.name] = learned_words
        print(f"\n{category.name} Learned Words: {', '.join(learned_words)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"test_results/word_learning_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nWord learning test results saved to test_results/word_learning_{timestamp}.json")

def main():
    """Run all language development tests"""
    print("Starting Language Development System Tests...")
    
    # Create test results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Run tests
    test_language_progression()
    test_word_learning()
    
    print("\nAll language development tests completed!")

if __name__ == "__main__":
    main() 