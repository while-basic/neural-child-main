import os

def create_file(filename, content):
    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

# Create main.py
main_content = '''# main.py
# Description: Main controller for the neural child development system
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from enum import Enum, auto
from typing import Dict, List, Optional
from datetime import datetime
import json
from conversation_system import ConversationSystem
from logger import DevelopmentLogger
from memory_store import MemoryStore
from language_development import LanguageDevelopment, LanguageStage

class DevelopmentalStage(Enum):
    NEWBORN = auto()         # 0-3 months
    INFANT = auto()          # 3-6 months
    EARLY_TODDLER = auto()   # 6-12 months
    LATE_TODDLER = auto()    # 12-18 months
    EARLY_PRESCHOOL = auto() # 18-24 months
    LATE_PRESCHOOL = auto()  # 2-3 years
    EARLY_CHILDHOOD = auto() # 3-4 years
    MIDDLE_CHILDHOOD = auto() # 4-5 years
    LATE_CHILDHOOD = auto()  # 5-6 years

class CurriculumManager:
    def __init__(self):
        self.current_stage = DevelopmentalStage.NEWBORN
        self.stage_progress = 0.0
        self.stage_durations = {
            DevelopmentalStage.NEWBORN: 3,        # 3 months
            DevelopmentalStage.INFANT: 3,         # 3 months
            DevelopmentalStage.EARLY_TODDLER: 6,  # 6 months
            DevelopmentalStage.LATE_TODDLER: 6,   # 6 months
            DevelopmentalStage.EARLY_PRESCHOOL: 6,# 6 months
            DevelopmentalStage.LATE_PRESCHOOL: 12,# 12 months
            DevelopmentalStage.EARLY_CHILDHOOD: 12,# 12 months
            DevelopmentalStage.MIDDLE_CHILDHOOD: 12,# 12 months
            DevelopmentalStage.LATE_CHILDHOOD: 12 # 12 months
        }
        self.start_time = datetime.now()
        
    def set_stage(self, stage: DevelopmentalStage, progress: float = 0.0):
        """Manually set the developmental stage and progress"""
        self.current_stage = stage
        self.stage_progress = max(0.0, min(1.0, progress))
        
    def update_progress(self, delta: float):
        """Update progress in current stage"""
        self.stage_progress += delta
        if self.stage_progress >= 1.0:
            self._advance_stage()
    
    def _advance_stage(self):
        """Advance to next developmental stage"""
        stages = list(DevelopmentalStage)
        current_idx = stages.index(self.current_stage)
        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            self.stage_progress = 0.0

class MotherLLM:
    def __init__(self):
        self.conversation_system = None
        
    def process_child_response(self, response: str, emotional_state: torch.Tensor) -> Dict:
        """Process child's response and provide guidance"""
        if self.conversation_system:
            return self.conversation_system.process_mother_response(response, emotional_state)
        return {}

class DigitalChild:
    def __init__(self, initial_stage: DevelopmentalStage = DevelopmentalStage.NEWBORN):
        # Initialize curriculum and development tracking
        self.curriculum = CurriculumManager()
        self.curriculum.current_stage = initial_stage
        
        # Initialize neural components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.brain = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(self.device)
        
        # Initialize emotional state: [joy, trust, fear, surprise]
        self.emotional_state = torch.tensor([0.5, 0.5, 0.2, 0.3], device=self.device)
        
        # Initialize language development
        self.language = LanguageDevelopment(device=self.device)
        
        # Initialize mother LLM
        self.mother = MotherLLM()
        
        # Initialize conversation system
        self.conversation_system = None
        
        # Track development metrics
        self.metrics = {
            'learning_rate': 0.1,
            'attention_span': 0.2,
            'memory_retention': 0.3,
            'social_awareness': 0.1,
            'emotional_regulation': 0.2
        }
        
        # Development history
        self.development_history = []
        
    def set_stage(self, stage: DevelopmentalStage, progress: float = 0.0):
        """Set the developmental stage of the child"""
        self.curriculum.set_stage(stage, progress)
        
        # Update language stage based on developmental stage
        stage_to_language = {
            DevelopmentalStage.NEWBORN: LanguageStage.PRELINGUISTIC,
            DevelopmentalStage.INFANT: LanguageStage.PRELINGUISTIC,
            DevelopmentalStage.EARLY_TODDLER: LanguageStage.HOLOPHRASTIC,
            DevelopmentalStage.LATE_TODDLER: LanguageStage.TELEGRAPHIC,
            DevelopmentalStage.EARLY_PRESCHOOL: LanguageStage.MULTIWORD,
            DevelopmentalStage.LATE_PRESCHOOL: LanguageStage.COMPLEX,
            DevelopmentalStage.EARLY_CHILDHOOD: LanguageStage.COMPLEX,
            DevelopmentalStage.MIDDLE_CHILDHOOD: LanguageStage.COMPLEX,
            DevelopmentalStage.LATE_CHILDHOOD: LanguageStage.COMPLEX
        }
        
        self.language.current_stage = stage_to_language[stage]
        self.language.stage_progress = progress
        
        # Update metrics based on stage
        self.metrics['learning_rate'] = min(0.9, 0.1 + (progress * 0.4))
        self.metrics['attention_span'] = min(0.9, 0.2 + (progress * 0.5))
        self.metrics['memory_retention'] = min(0.9, 0.3 + (progress * 0.4))
        self.metrics['social_awareness'] = min(0.9, 0.1 + (progress * 0.6))
        self.metrics['emotional_regulation'] = min(0.9, 0.2 + (progress * 0.5))
        
        # Record development milestone
        self.development_history.append({
            'timestamp': datetime.now().isoformat(),
            'stage': stage.name,
            'progress': progress,
            'metrics': self.metrics.copy()
        })
    
    def age(self) -> int:
        """Calculate age in months based on developmental stage"""
        total_months = 0
        stages = list(DevelopmentalStage)
        current_idx = stages.index(self.curriculum.current_stage)
        
        # Add months from completed stages
        for i in range(current_idx):
            total_months += self.curriculum.stage_durations[stages[i]]
        
        # Add months from current stage progress
        total_months += int(self.curriculum.stage_durations[self.curriculum.current_stage] 
                          * self.curriculum.stage_progress)
        
        return total_months
    
    def update_emotions(self, emotion_vector: torch.Tensor):
        """Update emotional state"""
        # Ensure emotion vector is on the correct device
        emotion_vector = emotion_vector.to(self.device)
        
        # Update emotional state with some momentum
        self.emotional_state = 0.7 * self.emotional_state + 0.3 * emotion_vector
        
        # Normalize emotions to be between 0 and 1
        self.emotional_state = torch.clamp(self.emotional_state, 0, 1)
    
    def express_feeling(self) -> str:
        """Express current emotional state"""
        joy, trust, fear, surprise = self.emotional_state.cpu().numpy()
        
        # Determine dominant emotion
        emotions = {'JOY': joy, 'TRUST': trust, 'FEAR': fear, 'SURPRISE': surprise}
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        # Return emotional state description
        if dominant_emotion[1] > 0.7:
            return f"[HIGH {dominant_emotion[0]}]"
        elif dominant_emotion[1] > 0.4:
            return f"[MODERATE {dominant_emotion[0]}]"
        else:
            return "[NEUTRAL]"
    
    def save_state(self, filepath: str):
        """Save current state to file"""
        state = {
            'stage': self.curriculum.current_stage.name,
            'stage_progress': float(self.curriculum.stage_progress),
            'emotional_state': self.emotional_state.cpu().tolist(),
            'metrics': self.metrics,
            'development_history': self.development_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.curriculum.current_stage = DevelopmentalStage[state['stage']]
        self.curriculum.stage_progress = state['stage_progress']
        self.emotional_state = torch.tensor(state['emotional_state'], device=self.device)
        self.metrics = state['metrics']
        self.development_history = state['development_history']'''

# Create chat.py
chat_content = r'''# chat.py
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
    main()'''

create_file('main.py', main_content)
print("Created main.py with UTF-8 encoding")

create_file('chat.py', chat_content)
print("Created chat.py with UTF-8 encoding") 