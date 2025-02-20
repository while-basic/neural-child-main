# main.py
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
        self.development_history = state['development_history']