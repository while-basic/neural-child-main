# main.py
# Description: Main controller for the neural child development system
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import networkx as nx
from pathlib import Path
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
    PRE_ADOLESCENT = auto()  # 6-12 years
    EARLY_TEEN = auto()      # 12-14 years
    MID_TEEN = auto()        # 14-16 years
    LATE_TEEN = auto()       # 16-18 years
    YOUNG_ADULT = auto()     # 18-21 years
    EARLY_TWENTIES = auto()  # 21-25 years
    LATE_TWENTIES = auto()   # 25-30 years

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 256, hidden_sizes: List[int] = [512, 256, 128]):
        super().__init__()
        
        # Store layer info for visualization
        self.layer_info = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'connections': []
        }
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            
        # Update connections after layers are created
        self.layer_info['connections'] = self._get_layer_connections()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def _get_layer_connections(self) -> List[Dict[str, Any]]:
        """Get neural network layer connections for visualization."""
        connections = []
        sizes = [self.layer_info['input_size']] + self.layer_info['hidden_sizes']
        
        for i in range(len(sizes)-1):
            connections.append({
                'from_layer': i,
                'to_layer': i+1,
                'from_size': sizes[i],
                'to_size': sizes[i+1],
                'type': 'feedforward'
            })
            
        return connections

class CurriculumManager:
    def __init__(self):
        self.current_stage = DevelopmentalStage.NEWBORN
        self.stage_progress = 0.0
        self.stage_durations = {
            DevelopmentalStage.NEWBORN: 3,         # 3 months
            DevelopmentalStage.INFANT: 3,          # 3 months
            DevelopmentalStage.EARLY_TODDLER: 6,   # 6 months
            DevelopmentalStage.LATE_TODDLER: 6,    # 6 months
            DevelopmentalStage.EARLY_PRESCHOOL: 6, # 6 months
            DevelopmentalStage.LATE_PRESCHOOL: 12, # 12 months
            DevelopmentalStage.EARLY_CHILDHOOD: 12,# 12 months
            DevelopmentalStage.MIDDLE_CHILDHOOD: 12,# 12 months
            DevelopmentalStage.LATE_CHILDHOOD: 12, # 12 months
            DevelopmentalStage.PRE_ADOLESCENT: 72, # 72 months (6 years)
            DevelopmentalStage.EARLY_TEEN: 24,     # 24 months (2 years)
            DevelopmentalStage.MID_TEEN: 24,       # 24 months (2 years)
            DevelopmentalStage.LATE_TEEN: 24,      # 24 months (2 years)
            DevelopmentalStage.YOUNG_ADULT: 36,    # 36 months (3 years)
            DevelopmentalStage.EARLY_TWENTIES: 48, # 48 months (4 years)
            DevelopmentalStage.LATE_TWENTIES: 60   # 60 months (5 years)
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
        self.brain = NeuralNetwork().to(self.device)
        
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
        
        # Self-concept and age awareness
        self.self_concept = {
            'age_range': self._get_age_range(initial_stage),
            'developmental_stage': initial_stage,
            'identity': self._get_identity_for_stage(initial_stage),
            'capabilities': self._get_capabilities_for_stage(initial_stage)
        }
        
    def _get_age_range(self, stage: DevelopmentalStage) -> str:
        """Get the age range for a given developmental stage"""
        age_ranges = {
            DevelopmentalStage.NEWBORN: "0-3 months",
            DevelopmentalStage.INFANT: "3-6 months",
            DevelopmentalStage.EARLY_TODDLER: "6-12 months",
            DevelopmentalStage.LATE_TODDLER: "12-18 months",
            DevelopmentalStage.EARLY_PRESCHOOL: "18-24 months",
            DevelopmentalStage.LATE_PRESCHOOL: "2-3 years",
            DevelopmentalStage.EARLY_CHILDHOOD: "3-4 years",
            DevelopmentalStage.MIDDLE_CHILDHOOD: "4-5 years",
            DevelopmentalStage.LATE_CHILDHOOD: "5-6 years",
            DevelopmentalStage.PRE_ADOLESCENT: "6-12 years",
            DevelopmentalStage.EARLY_TEEN: "12-14 years",
            DevelopmentalStage.MID_TEEN: "14-16 years",
            DevelopmentalStage.LATE_TEEN: "16-18 years",
            DevelopmentalStage.YOUNG_ADULT: "18-21 years",
            DevelopmentalStage.EARLY_TWENTIES: "21-25 years",
            DevelopmentalStage.LATE_TWENTIES: "25-30 years"
        }
        return age_ranges.get(stage, "unknown")
        
    def _get_identity_for_stage(self, stage: DevelopmentalStage) -> Dict[str, Any]:
        """Get identity characteristics for a given developmental stage"""
        identities = {
            DevelopmentalStage.NEWBORN: {
                'self_reference': "baby",
                'awareness_level': "minimal",
                'social_role': "infant"
            },
            DevelopmentalStage.INFANT: {
                'self_reference': "baby",
                'awareness_level': "basic",
                'social_role': "infant"
            },
            DevelopmentalStage.EARLY_TODDLER: {
                'self_reference': "baby",
                'awareness_level': "emerging",
                'social_role': "toddler"
            },
            DevelopmentalStage.LATE_TODDLER: {
                'self_reference': "toddler",
                'awareness_level': "developing",
                'social_role': "toddler"
            },
            DevelopmentalStage.EARLY_PRESCHOOL: {
                'self_reference': "kid",
                'awareness_level': "basic self-aware",
                'social_role': "preschooler"
            },
            DevelopmentalStage.LATE_PRESCHOOL: {
                'self_reference': "kid",
                'awareness_level': "self-aware",
                'social_role': "preschooler"
            },
            DevelopmentalStage.EARLY_CHILDHOOD: {
                'self_reference': "kid",
                'awareness_level': "clearly self-aware",
                'social_role': "young student"
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD: {
                'self_reference': "kid",
                'awareness_level': "fully self-aware",
                'social_role': "student"
            },
            DevelopmentalStage.LATE_CHILDHOOD: {
                'self_reference': "kid",
                'awareness_level': "socially aware",
                'social_role': "older student"
            },
            DevelopmentalStage.PRE_ADOLESCENT: {
                'self_reference': "preteen",
                'awareness_level': "highly self-aware",
                'social_role': "preteen student"
            },
            DevelopmentalStage.EARLY_TEEN: {
                'self_reference': "teenager",
                'awareness_level': "identity forming",
                'social_role': "young teen"
            },
            DevelopmentalStage.MID_TEEN: {
                'self_reference': "teenager",
                'awareness_level': "identity exploring",
                'social_role': "teenager"
            },
            DevelopmentalStage.LATE_TEEN: {
                'self_reference': "teenager",
                'awareness_level': "identity consolidating",
                'social_role': "older teen"
            },
            DevelopmentalStage.YOUNG_ADULT: {
                'self_reference': "young adult",
                'awareness_level': "fully formed identity",
                'social_role': "young professional"
            },
            DevelopmentalStage.EARLY_TWENTIES: {
                'self_reference': "adult",
                'awareness_level': "mature identity",
                'social_role': "professional"
            },
            DevelopmentalStage.LATE_TWENTIES: {
                'self_reference': "adult",
                'awareness_level': "established identity",
                'social_role': "experienced professional"
            }
        }
        return identities.get(stage, {})
        
    def _get_capabilities_for_stage(self, stage: DevelopmentalStage) -> Dict[str, Any]:
        """Get capabilities for a given developmental stage"""
        capabilities = {
            DevelopmentalStage.NEWBORN: {
                'cognitive': "basic sensory processing",
                'physical': "minimal motor control",
                'social': "basic emotional responses",
                'interests': ["bright colors", "faces", "sounds"]
            },
            DevelopmentalStage.INFANT: {
                'cognitive': "sensory exploration",
                'physical': "reaching and grasping",
                'social': "social smiling",
                'interests': ["faces", "objects", "sounds", "movement"]
            },
            DevelopmentalStage.EARLY_TODDLER: {
                'cognitive': "object permanence",
                'physical': "crawling",
                'social': "stranger anxiety",
                'interests': ["exploration", "simple toys", "peek-a-boo"]
            },
            DevelopmentalStage.LATE_TODDLER: {
                'cognitive': "symbolic thought",
                'physical': "walking",
                'social': "parallel play",
                'interests': ["independence", "simple games", "exploration"]
            },
            DevelopmentalStage.EARLY_PRESCHOOL: {
                'cognitive': "magical thinking",
                'physical': "running and climbing",
                'social': "imaginative play",
                'interests': ["pretend play", "stories", "art"]
            },
            DevelopmentalStage.LATE_PRESCHOOL: {
                'cognitive': "basic problem solving",
                'physical': "refined motor skills",
                'social': "cooperative play",
                'interests': ["games", "friends", "creative activities"]
            },
            DevelopmentalStage.EARLY_CHILDHOOD: {
                'cognitive': "concrete operations",
                'physical': "sports skills",
                'social': "friendship formation",
                'interests': ["sports", "games", "learning"]
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD: {
                'cognitive': "logical thinking",
                'physical': "athletic skills",
                'social': "peer relationships",
                'interests': ["hobbies", "friends", "achievements"]
            },
            DevelopmentalStage.LATE_CHILDHOOD: {
                'cognitive': "abstract thinking beginning",
                'physical': "refined coordination",
                'social': "group dynamics",
                'interests': ["complex games", "social activities", "learning"]
            },
            DevelopmentalStage.PRE_ADOLESCENT: {
                'cognitive': "abstract reasoning",
                'physical': "puberty beginning",
                'social': "peer importance",
                'interests': ["social media", "hobbies", "independence"]
            },
            DevelopmentalStage.EARLY_TEEN: {
                'cognitive': "complex reasoning",
                'physical': "puberty progression",
                'social': "identity exploration",
                'interests': ["social life", "identity", "future"]
            },
            DevelopmentalStage.MID_TEEN: {
                'cognitive': "advanced reasoning",
                'physical': "physical maturation",
                'social': "romantic interests",
                'interests': ["relationships", "future plans", "independence"]
            },
            DevelopmentalStage.LATE_TEEN: {
                'cognitive': "formal operations",
                'physical': "physical maturity",
                'social': "intimate relationships",
                'interests': ["career", "education", "relationships"]
            },
            DevelopmentalStage.YOUNG_ADULT: {
                'cognitive': "mature thinking",
                'physical': "peak physical condition",
                'social': "adult relationships",
                'interests': ["career", "relationships", "personal growth"]
            },
            DevelopmentalStage.EARLY_TWENTIES: {
                'cognitive': "expertise development",
                'physical': "maintained fitness",
                'social': "professional networking",
                'interests': ["career advancement", "relationships", "life goals"]
            },
            DevelopmentalStage.LATE_TWENTIES: {
                'cognitive': "mastery",
                'physical': "optimal health maintenance",
                'social': "established relationships",
                'interests': ["career mastery", "family planning", "life balance"]
            }
        }
        return capabilities.get(stage, {})

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
            DevelopmentalStage.LATE_CHILDHOOD: LanguageStage.COMPLEX,
            DevelopmentalStage.PRE_ADOLESCENT: LanguageStage.ADVANCED,
            DevelopmentalStage.EARLY_TEEN: LanguageStage.ADVANCED,
            DevelopmentalStage.MID_TEEN: LanguageStage.ADVANCED,
            DevelopmentalStage.LATE_TEEN: LanguageStage.ADVANCED,
            DevelopmentalStage.YOUNG_ADULT: LanguageStage.MASTERY,
            DevelopmentalStage.EARLY_TWENTIES: LanguageStage.MASTERY,
            DevelopmentalStage.LATE_TWENTIES: LanguageStage.MASTERY
        }
        
        self.language.current_stage = stage_to_language[stage]
        self.language.stage_progress = progress
        
        # Update self-concept for the new stage
        self.self_concept = {
            'age_range': self._get_age_range(stage),
            'developmental_stage': stage,
            'identity': self._get_identity_for_stage(stage),
            'capabilities': self._get_capabilities_for_stage(stage)
        }
        
        # Base metrics for each developmental stage
        stage_base_metrics = {
            DevelopmentalStage.NEWBORN: {
                'learning_rate': 0.1,
                'attention_span': 0.1,
                'memory_retention': 0.1,
                'social_awareness': 0.05,
                'emotional_regulation': 0.05
            },
            DevelopmentalStage.INFANT: {
                'learning_rate': 0.2,
                'attention_span': 0.15,
                'memory_retention': 0.15,
                'social_awareness': 0.1,
                'emotional_regulation': 0.1
            },
            DevelopmentalStage.EARLY_TODDLER: {
                'learning_rate': 0.3,
                'attention_span': 0.2,
                'memory_retention': 0.2,
                'social_awareness': 0.15,
                'emotional_regulation': 0.15
            },
            DevelopmentalStage.LATE_TODDLER: {
                'learning_rate': 0.4,
                'attention_span': 0.25,
                'memory_retention': 0.25,
                'social_awareness': 0.2,
                'emotional_regulation': 0.2
            },
            DevelopmentalStage.EARLY_PRESCHOOL: {
                'learning_rate': 0.5,
                'attention_span': 0.3,
                'memory_retention': 0.3,
                'social_awareness': 0.25,
                'emotional_regulation': 0.25
            },
            DevelopmentalStage.LATE_PRESCHOOL: {
                'learning_rate': 0.6,
                'attention_span': 0.4,
                'memory_retention': 0.4,
                'social_awareness': 0.3,
                'emotional_regulation': 0.3
            },
            DevelopmentalStage.EARLY_CHILDHOOD: {
                'learning_rate': 0.7,
                'attention_span': 0.5,
                'memory_retention': 0.5,
                'social_awareness': 0.4,
                'emotional_regulation': 0.4
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD: {
                'learning_rate': 0.75,
                'attention_span': 0.6,
                'memory_retention': 0.6,
                'social_awareness': 0.5,
                'emotional_regulation': 0.5
            },
            DevelopmentalStage.LATE_CHILDHOOD: {
                'learning_rate': 0.8,
                'attention_span': 0.7,
                'memory_retention': 0.7,
                'social_awareness': 0.6,
                'emotional_regulation': 0.6
            },
            DevelopmentalStage.PRE_ADOLESCENT: {
                'learning_rate': 0.85,
                'attention_span': 0.75,
                'memory_retention': 0.75,
                'social_awareness': 0.7,
                'emotional_regulation': 0.65
            },
            DevelopmentalStage.EARLY_TEEN: {
                'learning_rate': 0.9,
                'attention_span': 0.8,
                'memory_retention': 0.8,
                'social_awareness': 0.75,
                'emotional_regulation': 0.7
            },
            DevelopmentalStage.MID_TEEN: {
                'learning_rate': 0.9,
                'attention_span': 0.85,
                'memory_retention': 0.85,
                'social_awareness': 0.8,
                'emotional_regulation': 0.75
            },
            DevelopmentalStage.LATE_TEEN: {
                'learning_rate': 0.9,
                'attention_span': 0.9,
                'memory_retention': 0.9,
                'social_awareness': 0.85,
                'emotional_regulation': 0.8
            },
            DevelopmentalStage.YOUNG_ADULT: {
                'learning_rate': 0.95,
                'attention_span': 0.95,
                'memory_retention': 0.95,
                'social_awareness': 0.9,
                'emotional_regulation': 0.85
            },
            DevelopmentalStage.EARLY_TWENTIES: {
                'learning_rate': 0.95,
                'attention_span': 0.95,
                'memory_retention': 0.95,
                'social_awareness': 0.95,
                'emotional_regulation': 0.9
            },
            DevelopmentalStage.LATE_TWENTIES: {
                'learning_rate': 1.0,
                'attention_span': 1.0,
                'memory_retention': 1.0,
                'social_awareness': 1.0,
                'emotional_regulation': 1.0
            }
        }
        
        # Get base metrics for current stage
        base_metrics = stage_base_metrics[stage]
        
        # Update metrics with base values plus progress-based improvement
        for metric, base_value in base_metrics.items():
            self.metrics[metric] = min(1.0, base_value + (progress * 0.1))
        
        # Record development milestone
        self.development_history.append({
            'timestamp': datetime.now().isoformat(),
            'stage': stage.name,
            'progress': progress,
            'metrics': self.metrics.copy(),
            'self_concept': self.self_concept.copy()
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

    def get_network_structure(self) -> Dict[str, Any]:
        """Get neural network structure for visualization."""
        return {
            'brain': {
                'type': 'feedforward',
                'layers': self.brain.layer_info,
                'metrics': {
                    'learning_rate': self.metrics['learning_rate'],
                    'memory_retention': self.metrics['memory_retention']
                }
            },
            'language': {
                'type': 'language_network',
                'stage': self.language.current_stage.name,
                'vocabulary_size': len(self.language.vocabulary),
                'connections': self._get_language_connections()
            },
            'emotional': {
                'type': 'emotional_network',
                'state': self.emotional_state.cpu().tolist(),
                'regulation': self.metrics['emotional_regulation']
            },
            'development': {
                'stage': self.curriculum.current_stage.name,
                'age_months': self.age(),
                'metrics': self.metrics
            }
        }
        
    def _get_language_connections(self) -> List[Dict[str, Any]]:
        """Get language network connections."""
        connections = []
        
        # Add connections between words in the vocabulary
        for category, words in self.language.vocabulary.items():
            for word in words:
                connections.append({
                    'type': 'vocabulary',
                    'category': category.name,
                    'word': word,
                    'stage': self.language.current_stage.name
                })
                
        return connections
        
    def export_network_visualization(self, obsidian_path: Path) -> None:
        """Export network visualization to Obsidian."""
        network_dir = obsidian_path / "Network"
        network_dir.mkdir(parents=True, exist_ok=True)
        
        # Create network visualization file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        network_data = self.get_network_structure()
        
        content = f'''# Neural Network Structure: {timestamp}

## Brain Network
- Type: {network_data['brain']['type']}
- Input Size: {network_data['brain']['layers']['input_size']}
- Hidden Layers: {network_data['brain']['layers']['hidden_sizes']}
- Learning Rate: {network_data['brain']['metrics']['learning_rate']:.2f}
- Memory Retention: {network_data['brain']['metrics']['memory_retention']:.2f}

### Layer Connections
```mermaid
graph TD
    Input[Input Layer ({network_data['brain']['layers']['input_size']} neurons)]
'''
        
        # Add hidden layers to diagram
        for i, size in enumerate(network_data['brain']['layers']['hidden_sizes']):
            content += f"    H{i}[Hidden Layer {i+1} ({size} neurons)]\n"
            if i == 0:
                content += f"    Input --> H{i}\n"
            else:
                content += f"    H{i-1} --> H{i}\n"
                
        content += "```\n\n"
        
        # Add language network
        content += f'''## Language Network
- Stage: {network_data['language']['stage']}
- Vocabulary Size: {network_data['language']['vocabulary_size']}

### Word Connections
```mermaid
graph LR
'''
        
        # Add word connections
        for conn in network_data['language']['connections'][:10]:  # Limit to 10 for readability
            content += f"    {conn['word']}[{conn['word']}] --> {conn['category']}[{conn['category']}]\n"
            
        content += "```\n\n"
        
        # Add emotional network
        content += f'''## Emotional Network
- Current State:
  - Joy: {network_data['emotional']['state'][0]:.2f}
  - Trust: {network_data['emotional']['state'][1]:.2f}
  - Fear: {network_data['emotional']['state'][2]:.2f}
  - Surprise: {network_data['emotional']['state'][3]:.2f}
- Regulation: {network_data['emotional']['regulation']:.2f}

### Emotional Connections
```mermaid
graph TD
    Joy[Joy] --> Trust[Trust]
    Trust --> Fear[Fear]
    Fear --> Surprise[Surprise]
    Surprise --> Joy
```

## Development Status
- Stage: {network_data['development']['stage']}
- Age: {network_data['development']['age_months']} months
- Metrics:
  - Learning Rate: {network_data['development']['metrics']['learning_rate']:.2f}
  - Attention Span: {network_data['development']['metrics']['attention_span']:.2f}
  - Memory Retention: {network_data['development']['metrics']['memory_retention']:.2f}
  - Social Awareness: {network_data['development']['metrics']['social_awareness']:.2f}
  - Emotional Regulation: {network_data['development']['metrics']['emotional_regulation']:.2f}
'''
        
        # Write visualization file
        viz_file = network_dir / f"network_structure_{timestamp}.md"
        with open(viz_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Update README with link to visualization
        readme_path = network_dir / "README.md"
        if not readme_path.exists():
            readme_content = f'''# Neural Network Visualizations

This directory contains visualizations of the neural child's brain structure and network connections.

## Latest Visualization
- [[network_structure_{timestamp}|Network Structure {timestamp}]]

## Network Components
- Brain Network (feedforward neural network)
- Language Network (vocabulary and concepts)
- Emotional Network (emotional states and regulation)
- Development Status (stages and metrics)
'''
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)