# test_development.py
# Description: Test script for the neural child development system
# Created by: Christopher Celaya

import torch
from main import DigitalChild, DevelopmentalStage
from language_development import LanguageStage
import json
from datetime import datetime

def simulate_vocabulary_growth(child, num_words=50):
    """Simulate vocabulary growth by learning words appropriate to the stage"""
    stage_appropriate_words = {
        LanguageStage.PRELINGUISTIC: [
            "mama", "dada", "baba", "milk", "up", "no"
        ],
        LanguageStage.HOLOPHRASTIC: [
            "ball", "dog", "cat", "eat", "sleep", "play", "more", "want",
            "big", "hot", "cold", "good"
        ],
        LanguageStage.TELEGRAPHIC: [
            "car", "book", "toy", "run", "jump", "walk", "red", "blue",
            "happy", "sad", "hungry", "thirsty"
        ],
        LanguageStage.MULTIWORD: [
            "please", "thank", "help", "like", "love", "give", "take",
            "pretty", "fast", "slow", "nice", "bad"
        ],
        LanguageStage.COMPLEX: [
            "beautiful", "interesting", "exciting", "wonderful", "difficult",
            "understand", "remember", "forget", "think", "feel"
        ],
        LanguageStage.ADVANCED: [
            "philosophical", "theoretical", "analytical", "comprehensive",
            "investigate", "hypothesize", "conclude", "interpret"
        ],
        LanguageStage.MASTERY: [
            "epistemological", "paradigmatic", "methodological", "empirical",
            "synthesize", "conceptualize", "extrapolate", "correlate"
        ]
    }
    
    # Get words for current stage
    current_stage = child.language.current_stage
    words_to_learn = stage_appropriate_words.get(current_stage, [])
    
    # Learn words
    for word in words_to_learn[:num_words]:
        child.language.learn_word(word)
    
    # Learn some concepts
    stage_concepts = {
        LanguageStage.MULTIWORD: [
            "playing together",
            "reading books",
            "helping others"
        ],
        LanguageStage.COMPLEX: [
            "sharing with friends",
            "solving problems",
            "learning new things"
        ],
        LanguageStage.ADVANCED: [
            "critical thinking",
            "abstract reasoning",
            "scientific method"
        ],
        LanguageStage.MASTERY: [
            "cognitive development",
            "theoretical framework",
            "empirical analysis"
        ]
    }
    
    # Learn concepts appropriate for the stage
    if current_stage in stage_concepts:
        for concept in stage_concepts[current_stage]:
            child.language.learn_concept(concept)

def test_development_progression():
    """Test the progression through developmental stages"""
    print("\n=== Testing Neural Child Development System ===\n")
    
    # Initialize digital child
    child = DigitalChild(initial_stage=DevelopmentalStage.NEWBORN)
    
    # Test progression through all stages
    stages = list(DevelopmentalStage)
    
    for stage in stages:
        # Set the stage
        print(f"\nTesting Stage: {stage.name}")
        child.set_stage(stage, progress=0.0)
        
        # Get current language stage
        lang_stage = child.language.current_stage
        print(f"Language Stage: {lang_stage.name}")
        
        # Simulate vocabulary growth
        simulate_vocabulary_growth(child)
        
        # Test emotional states
        emotional_state = torch.tensor([0.6, 0.5, 0.2, 0.3])  # [joy, trust, fear, surprise]
        
        # Generate some utterances
        contexts = [
            "Playing with toys and learning new things",
            "Discussing interesting ideas",
            "Solving complex problems"
        ]
        
        print("\nSample Utterances:")
        for context in contexts:
            utterance = child.language.generate_utterance(context, emotional_state)
            print(f"Context: {context}")
            print(f"Response: {utterance}\n")
        
        # Get development metrics
        metrics = child.metrics
        print("Development Metrics:")
        print(f"Learning Rate: {metrics['learning_rate']:.2f}")
        print(f"Attention Span: {metrics['attention_span']:.2f}")
        print(f"Memory Retention: {metrics['memory_retention']:.2f}")
        print(f"Social Awareness: {metrics['social_awareness']:.2f}")
        print(f"Emotional Regulation: {metrics['emotional_regulation']:.2f}")
        
        # Test language development
        lang_summary = child.language.get_development_summary()
        print("\nLanguage Development:")
        print(f"Vocabulary Size: {lang_summary['vocabulary_size']}")
        print(f"Stage Progress: {lang_summary['stage_progress']:.2f}")
        
        # Test progression
        child.set_stage(stage, progress=1.0)
        print("\nAfter Learning:")
        print(f"Stage Progress: {child.curriculum.stage_progress:.2f}")
        
        print("\n" + "="*50)

def test_advanced_language_capabilities():
    """Test advanced and mastery language capabilities"""
    print("\n=== Testing Advanced Language Capabilities ===\n")
    
    # Initialize child at advanced stage
    child = DigitalChild(initial_stage=DevelopmentalStage.YOUNG_ADULT)
    
    # Simulate development through previous stages to build vocabulary
    for stage in [
        DevelopmentalStage.NEWBORN,
        DevelopmentalStage.INFANT,
        DevelopmentalStage.EARLY_TODDLER,
        DevelopmentalStage.LATE_TODDLER,
        DevelopmentalStage.EARLY_PRESCHOOL,
        DevelopmentalStage.LATE_PRESCHOOL,
        DevelopmentalStage.EARLY_CHILDHOOD,
        DevelopmentalStage.MIDDLE_CHILDHOOD,
        DevelopmentalStage.LATE_CHILDHOOD,
        DevelopmentalStage.PRE_ADOLESCENT,
        DevelopmentalStage.EARLY_TEEN,
        DevelopmentalStage.MID_TEEN,
        DevelopmentalStage.LATE_TEEN
    ]:
        child.set_stage(stage, progress=0.0)
        simulate_vocabulary_growth(child, num_words=20)
        child.set_stage(stage, progress=1.0)
    
    # Set final stage and develop advanced vocabulary
    child.set_stage(DevelopmentalStage.YOUNG_ADULT, progress=0.0)
    simulate_vocabulary_growth(child, num_words=100)
    
    # Test advanced language generation
    emotional_state = torch.tensor([0.7, 0.6, 0.1, 0.4])  # Positive emotional state
    contexts = [
        "Discussing philosophy and consciousness",
        "Writing a research paper on cognitive development",
        "Giving a presentation on neural networks",
        "Analyzing literature and linguistic patterns"
    ]
    
    print("Testing Advanced Language Generation:")
    for context in contexts:
        utterance = child.language.generate_utterance(context, emotional_state)
        print(f"\nContext: {context}")
        print(f"Generated Response: {utterance}")
    
    # Test language patterns
    print("\nTesting Language Patterns:")
    patterns = child.language.stage_patterns[LanguageStage.MASTERY]
    print("Mastery Stage Capabilities:")
    for category, capabilities in patterns.items():
        print(f"{category}: {capabilities}")
    
    # Test vocabulary and concepts
    lang_summary = child.language.get_development_summary()
    print("\nLanguage Development Summary:")
    print(f"Vocabulary Size: {lang_summary['vocabulary_size']}")
    print(f"Stage Progress: {lang_summary['stage_progress']:.2f}")
    
    if lang_summary['most_frequent_words']:
        print("\nMost Frequent Words:")
        for word, freq in lang_summary['most_frequent_words'][:5]:
            print(f"{word}: {freq}")
            
    # Test specific advanced capabilities
    print("\nTesting Specific Advanced Capabilities:")
    advanced_contexts = {
        "Philosophical Analysis": "The nature of consciousness raises fundamental questions about human experience and reality.",
        "Scientific Research": "Our methodology combines quantitative and qualitative approaches to ensure comprehensive results.",
        "Technical Presentation": "The neural network architecture implements attention mechanisms for improved performance.",
        "Literary Criticism": "The author's use of metaphor and symbolism creates a rich tapestry of meaning."
    }
    
    print("\nAdvanced Language Generation Examples:")
    for topic, context in advanced_contexts.items():
        print(f"\nTopic: {topic}")
        print(f"Context: {context}")
        utterance = child.language.generate_utterance(context, emotional_state)
        print(f"Response: {utterance}")

def main():
    """Main test function"""
    print("Starting Neural Child Development Tests...")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # Run basic development tests
        test_development_progression()
        
        # Run advanced language tests
        test_advanced_language_capabilities()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 