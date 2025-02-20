# dream_cycle_test.py
# Description: Test script for continuous dream cycle learning in the neural child development system
# Created by: Christopher Celaya

import torch
from main import DigitalChild, DevelopmentalStage
from chat_interface import ChatInterface
from vision_development import VisionDevelopment
from datetime import datetime, timedelta
import time
import random
import os
from pathlib import Path

def print_cycle_stats(cycle_num: int, vocab_stats: dict, emotional_context: torch.Tensor, stage: DevelopmentalStage, vision_metrics: dict = None):
    """Print statistics for the current dream cycle"""
    print(f"\n{'='*50}")
    print(f"Dream Cycle #{cycle_num} - Stage: {stage.name}")
    print(f"{'='*50}")
    print("\nVocabulary Stats:")
    print(f"Total Words: {vocab_stats['total_words']}")
    print(f"Obsidian Words: {vocab_stats['obsidian_words']}")
    print(f"ChromaDB Words: {vocab_stats['chroma_words']}")
    
    if vision_metrics:
        print("\nVision Development Metrics:")
        for metric, value in vision_metrics['metrics'].items():
            bar = "█" * int(value * 10)
            print(f"{metric:20}: {bar} {value:.3f}")
    
    print("\nEmotional State:")
    emotions = [
        "Joy", "Trust", "Fear", "Surprise", 
        "Sadness", "Anger", "Disgust", "Anticipation",
        "Love", "Guilt", "Hope", "Regret"
    ]
    for emotion, value in zip(emotions, emotional_context.tolist()):
        bar = "█" * int(value * 10)
        print(f"{emotion:12}: {bar} {value:.3f}")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def check_for_new_images(obsidian_path: Path, images_path: Path) -> list:
    """Check for new images in Obsidian vault and images folder"""
    image_files = []
    
    # Check Obsidian vault
    if obsidian_path.exists():
        for file in obsidian_path.rglob("*.{jpg,jpeg,png,gif}"):
            image_files.append(file)
            
    # Check images folder
    if images_path.exists():
        for file in images_path.rglob("*.{jpg,jpeg,png,gif}"):
            image_files.append(file)
            
    return image_files

def generate_image_based_dream(chat: ChatInterface, vision: VisionDevelopment, image_path: str) -> str:
    """Generate a dream based on image content and vision development"""
    # Process image through vision system
    vision_results = vision.process_image(image_path, chat.brain.stage)
    
    # Create dream prompt incorporating visual features
    prompt = f"""Based on what I can see in this image with my current visual capabilities:

Visual Development Stage: {chat.brain.stage.name}
Visual Acuity: {vision_results.get('visual_acuity', 0.0):.2f}
Color Perception: {vision_results.get('color_perception', 0.0):.2f}
Pattern Recognition: {vision_results.get('pattern_recognition', 0.0):.2f}
Object Recognition: {vision_results.get('object_permanence', 0.0):.2f}

Generate a child-like dream that incorporates what I can perceive from the image.
The dream should reflect my current developmental stage and emotional state.

Current Emotional State:
Joy: {float(chat.emotional_context[0]):.2f}
Trust: {float(chat.emotional_context[1]):.2f}
Fear: {float(chat.emotional_context[2]):.2f}
Surprise: {float(chat.emotional_context[3]):.2f}
Sadness: {float(chat.emotional_context[4]):.2f}
Anger: {float(chat.emotional_context[5]):.2f}
Disgust: {float(chat.emotional_context[6]):.2f}
Anticipation: {float(chat.emotional_context[7]):.2f}

Generate a dream narrative:"""

    # Get dream from Ollama
    response = chat.ollama.generate_response(
        message=prompt,
        emotional_state={
            'joy': float(chat.emotional_context[0]),
            'trust': float(chat.emotional_context[1]),
            'fear': float(chat.emotional_context[2]),
            'surprise': float(chat.emotional_context[3]),
            'sadness': float(chat.emotional_context[4]),
            'anger': float(chat.emotional_context[5]),
            'disgust': float(chat.emotional_context[6]),
            'anticipation': float(chat.emotional_context[7]),
            'love': float(chat.emotional_context[8]),
            'guilt': float(chat.emotional_context[9]),
            'hope': float(chat.emotional_context[10]),
            'regret': float(chat.emotional_context[11])
        },
        brain_state=chat.brain.get_brain_state(),
        stage=chat.brain.stage.name,
        age_months=chat._get_age_months()
    )
    
    return response

def generate_thoughts(chat: ChatInterface, dream: str, num_thoughts: int = 3) -> list:
    """Generate thoughts using Ollama based on dream content and vocabulary"""
    # Get recent vocabulary
    vocab_stats = chat.get_vocabulary_stats()
    known_words = list(chat.known_words)
    
    # Create thought generation prompt
    prompt = f"""Based on this dream and using some of these known words ({', '.join(known_words[:20])}), 
generate {num_thoughts} different child-like thoughts or reactions.
Each thought should be a simple sentence expressing a feeling, question, or desire.
The thoughts should be appropriate for developmental stage {chat.brain.stage.name} (age {chat._get_age_months()} months).

Recent Dream:
{dream}

Current Emotional State:
Joy: {float(chat.emotional_context[0]):.2f}
Trust: {float(chat.emotional_context[1]):.2f}
Fear: {float(chat.emotional_context[2]):.2f}
Surprise: {float(chat.emotional_context[3]):.2f}
Sadness: {float(chat.emotional_context[4]):.2f}
Anger: {float(chat.emotional_context[5]):.2f}
Disgust: {float(chat.emotional_context[6]):.2f}
Anticipation: {float(chat.emotional_context[7]):.2f}

Generate {num_thoughts} thoughts appropriate for this developmental stage, one per line:"""

    # Get thoughts from Ollama
    response = chat.ollama.generate_response(
        message=prompt,
        emotional_state={
            'joy': float(chat.emotional_context[0]),
            'trust': float(chat.emotional_context[1]),
            'fear': float(chat.emotional_context[2]),
            'surprise': float(chat.emotional_context[3]),
            'sadness': float(chat.emotional_context[4]),
            'anger': float(chat.emotional_context[5]),
            'disgust': float(chat.emotional_context[6]),
            'anticipation': float(chat.emotional_context[7]),
            'love': float(chat.emotional_context[8]),
            'guilt': float(chat.emotional_context[9]),
            'hope': float(chat.emotional_context[10]),
            'regret': float(chat.emotional_context[11])
        },
        brain_state=chat.brain.get_brain_state(),
        stage=chat.brain.stage.name,
        age_months=chat._get_age_months()
    )
    
    # Split response into individual thoughts
    thoughts = [t.strip() for t in response.split('\n') if t.strip()]
    return thoughts[:num_thoughts]  # Ensure we only return the requested number of thoughts

def run_developmental_stages(minutes_per_stage: int = 2, cycle_interval_seconds: int = 30):
    """Run dream cycles through all developmental stages"""
    print("Initializing Neural Child Development Stage Progression...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create digital child instance
    child = DigitalChild()
    
    # Create chat interface
    chat = ChatInterface(child.brain, device=device)
    
    # Initialize vision development
    vision = VisionDevelopment(device=device)
    
    # Setup paths for image monitoring
    obsidian_path = Path("obsidian_vault")
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)
    
    # Get all developmental stages
    stages = list(DevelopmentalStage)
    
    print("\nStarting Stage Progression...")
    print(f"Time per stage: {minutes_per_stage} minutes")
    print(f"Cycle Interval: {cycle_interval_seconds} seconds")
    print(f"Monitoring for images in: {obsidian_path} and {images_path}")
    
    start_time = datetime.now()
    total_cycles = 0
    processed_images = set()
    
    # Initial vocabulary stats
    initial_stats = chat.get_vocabulary_stats()
    
    try:
        # Progress through each stage
        for stage in stages:
            print(f"\n{'#'*70}")
            print(f"Entering Developmental Stage: {stage.name}")
            print(f"{'#'*70}")
            
            # Set the new stage
            child.set_stage(stage, progress=0.0)
            vision.current_stage = stage
            stage_start_time = datetime.now()
            stage_end_time = stage_start_time + timedelta(minutes=minutes_per_stage)
            stage_cycles = 0
            
            # Run cycles for this stage
            while datetime.now() < stage_end_time:
                total_cycles += 1
                stage_cycles += 1
                
                # Update stage progress based on time spent
                progress = min(1.0, (datetime.now() - stage_start_time).total_seconds() / 
                             (minutes_per_stage * 60))
                child.set_stage(stage, progress=progress)
                vision.stage_progress = progress
                
                # Check for new images
                new_images = check_for_new_images(obsidian_path, images_path)
                for image_path in new_images:
                    if str(image_path) not in processed_images:
                        print(f"\nProcessing new image: {image_path}")
                        # Generate dream based on image
                        dream = generate_image_based_dream(chat, vision, str(image_path))
                        processed_images.add(str(image_path))
                        print(dream)
                    
                # Generate regular dream if no new images
                if not new_images:
                    print(f"\nGenerating Dream #{total_cycles} (Stage {stage.name} - Progress: {progress:.2f})")
                    dream = chat._generate_dream()
                    print(dream)
                
                # Extract and learn new words from the dream
                new_words = chat._extract_new_words(dream)
                if new_words:
                    print(f"\nLearned {len(new_words)} new words from dream: {', '.join(new_words)}")
                    chat._learn_new_words(new_words)
                
                # Print cycle statistics
                stats = chat.get_vocabulary_stats()
                vision_metrics = vision.get_development_summary()
                print_cycle_stats(total_cycles, stats, chat.emotional_context, stage, vision_metrics)
                
                # Generate and process thoughts about the dream
                print("\nProcessing thoughts about the dream...")
                thoughts = generate_thoughts(chat, dream, num_thoughts=random.randint(2, 5))
                
                # Process each thought
                for thought in thoughts:
                    print(f"\nThought: {thought}")
                    chat.process_message(thought)
                
                # Wait for next cycle if needed
                if datetime.now() < stage_end_time and cycle_interval_seconds > 0:
                    print(f"\nSleeping for {cycle_interval_seconds} seconds...")
                    time.sleep(cycle_interval_seconds)
            
            print(f"\nCompleted {stage_cycles} cycles in {stage.name} stage")
            
    except KeyboardInterrupt:
        print("\nDevelopment progression interrupted by user")
    
    # Print final statistics
    final_stats = chat.get_vocabulary_stats()
    total_duration = datetime.now() - start_time
    
    print("\nFinal Development Results:")
    print(f"Total Stages Completed: {len(stages)}")
    print(f"Total Dream Cycles: {total_cycles}")
    print(f"Words Learned: {final_stats['total_words'] - initial_stats['total_words']}")
    print(f"Total Duration: {total_duration}")
    print(f"Final Stage: {child.curriculum.current_stage.name}")
    print_cycle_stats(total_cycles, final_stats, chat.emotional_context, child.curriculum.current_stage, 
                     vision.get_development_summary())

if __name__ == "__main__":
    # Run through all developmental stages with 5 minutes per stage
    run_developmental_stages(minutes_per_stage=5, cycle_interval_seconds=0) 