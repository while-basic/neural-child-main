# main.py
# Description: Main controller for the neural child development system
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import networkx as nx
from pathlib import Path
from conversation_system import ConversationSystem
from logger import DevelopmentLogger
from memory_store import MemoryStore
from language_development import LanguageDevelopment, LanguageStage
from emotional_memory_system import EmotionalMemorySystem, EmotionalMemoryEntry, EmotionalAssociation
from neural_architecture import NeuralArchitecture, BrainState, BrainRegion, CognitiveFunction

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
        
        # Stage progression requirements
        self.stage_requirements = {
            DevelopmentalStage.NEWBORN: {
                'interactions': 5,  # Number of interactions needed
                'emotional_range': 0.2,  # Range of emotions experienced
                'vocabulary_size': 3,  # Number of words learned
                'trust_level': 0.4  # Required trust level
            },
            DevelopmentalStage.INFANT: {
                'interactions': 10,
                'emotional_range': 0.3,
                'vocabulary_size': 8,
                'trust_level': 0.5
            },
            DevelopmentalStage.EARLY_TODDLER: {
                'interactions': 15,
                'emotional_range': 0.4,
                'vocabulary_size': 15,
                'trust_level': 0.6
            },
            DevelopmentalStage.LATE_TODDLER: {
                'interactions': 20,
                'emotional_range': 0.5,
                'vocabulary_size': 25,
                'trust_level': 0.6
            },
            DevelopmentalStage.EARLY_PRESCHOOL: {
                'interactions': 30,
                'emotional_range': 0.6,
                'vocabulary_size': 50,
                'trust_level': 0.7
            }
            # Add more stages as needed...
        }
        
        # Progress tracking
        self.interaction_count = 0
        self.unique_emotions = set()
        self.start_time = datetime.now()
        self.last_progress_check = datetime.now()
        
    def update_progress(self, interaction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update progress based on interaction data"""
        # Increment interaction count
        self.interaction_count += 1
        
        # Track emotional range
        if 'emotional_state' in interaction_data:
            emotions = interaction_data['emotional_state']
            for emotion, value in emotions.items():
                if value > 0.3:  # Only count significant emotions
                    self.unique_emotions.add(emotion)
        
        # Get requirements for current stage
        requirements = self.stage_requirements.get(self.current_stage, {})
        if not requirements:
            return None
            
        # Calculate progress metrics
        progress_metrics = {
            'interactions': min(1.0, self.interaction_count / requirements.get('interactions', 100)),
            'emotional_range': min(1.0, len(self.unique_emotions) / 4),  # 4 basic emotions
            'vocabulary_size': min(1.0, interaction_data.get('vocabulary_size', 0) / requirements.get('vocabulary_size', 100)),
            'trust_level': min(1.0, interaction_data.get('emotional_state', {}).get('trust', 0) / requirements.get('trust_level', 1.0))
        }
        
        # Calculate overall progress with weights
        weights = {
            'interactions': 0.3,
            'emotional_range': 0.3,
            'vocabulary_size': 0.2,
            'trust_level': 0.2
        }
        
        self.stage_progress = sum(
            metric * weights[name]
            for name, metric in progress_metrics.items()
        )
        
        # Check if enough time has passed since last progress check (at least 1 minute)
        time_since_last_check = (datetime.now() - self.last_progress_check).total_seconds()
        if time_since_last_check < 60:
            return None
            
        self.last_progress_check = datetime.now()
        
        # Check for stage advancement
        if self.stage_progress >= 1.0:
            return self._advance_stage(progress_metrics)
        
        return None
    
    def _advance_stage(self, progress_metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Advance to next developmental stage"""
        stages = list(DevelopmentalStage)
        current_idx = stages.index(self.current_stage)
        
        if current_idx < len(stages) - 1:
            old_stage = self.current_stage
            self.current_stage = stages[current_idx + 1]
            self.stage_progress = 0.0
            self.interaction_count = 0
            self.unique_emotions = set()
            
            # Return level up notification with detailed achievements
            return {
                'type': 'level_up',
                'old_stage': old_stage.name,
                'new_stage': self.current_stage.name,
                'message': f"🎉 Congratulations! Advanced from {old_stage.name} to {self.current_stage.name}!",
                'achievements': [
                    f"✨ Mastered {len(self.unique_emotions)} emotions ({progress_metrics['emotional_range']*100:.0f}% complete)",
                    f"📚 Learned {self.interaction_count} interactions ({progress_metrics['interactions']*100:.0f}% complete)",
                    f"🤝 Built trust to level {progress_metrics['trust_level']*100:.0f}%",
                    f"🔤 Vocabulary size: {progress_metrics['vocabulary_size']*100:.0f}% of goal"
                ]
            }
        
        return None

class MotherLLM:
    def __init__(self):
        self.conversation_system = None
        
    def process_child_response(self, response: str, emotional_state: torch.Tensor) -> Dict:
        """Process child's response and provide guidance"""
        if self.conversation_system:
            return self.conversation_system.process_mother_response(response, emotional_state)
        return {}

class BrainState:
    """Class to track brain state"""
    def __init__(self):
        self.emotional_valence = 0.0
        self.arousal = 0.0
        self.attention = 0.0
        self.consciousness = 0.0
        self.stress = 0.0
        self.fatigue = 0.0
        self.neurotransmitters = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'norepinephrine': 0.5,
            'gaba': 0.5,
            'glutamate': 0.5
        }

class IntegratedBrain(nn.Module):
    """Neural network-based brain that integrates all cognitive systems"""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, 
                 stage: DevelopmentalStage = DevelopmentalStage.NEWBORN):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.stage = stage
        
        # Initialize brain state
        self.brain_state = BrainState()
        
        # Get stage-specific modulation factors
        self.modulation_factors = self._get_stage_modulation(stage)
        
        # Sensory System - Processes visual and auditory input
        self.sensory_system = nn.ModuleDict({
            'visual': nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            ),
            'auditory': nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
        })
        
        # Memory System - Processes and integrates different types of memories
        self.memory_system = nn.ModuleDict({
            'working_memory': nn.LSTM(hidden_dim, hidden_dim, num_layers=2),
            'episodic_memory': nn.GRU(hidden_dim, hidden_dim, num_layers=2),
            'semantic_memory': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'memory_integration': nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        })
        
        # Memory smoothing factor
        self.memory_alpha = 0.7  # Controls memory persistence (higher = more persistent)
        self.last_memory = None
        
        # Emotional System - Processes and generates emotions
        self.emotional_system = nn.ModuleDict({
            'emotion_generator': nn.Sequential(
                nn.Linear(hidden_dim + 4, hidden_dim),  # Features + current emotions
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4)  # [joy, trust, fear, surprise]
            ),
            'emotion_modulator': nn.Sequential(
                nn.Linear(hidden_dim + 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        })
        
        # Decision Making System - Selects actions based on integrated information
        self.decision_system = nn.ModuleDict({
            'action_selector': nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            ),
            'response_generator': nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })
        
        # Learning System - Adjusts weights based on experience
        self.learning_system = nn.ModuleDict({
            'reinforcement': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # Reward prediction
            ),
            'adaptation': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })
        
        # Integration Network - Combines all subsystems
        total_features = hidden_dim * 3 + hidden_dim // 2  # sensory, memory, emotional, action
        self.integration_network = nn.Sequential(
            nn.Linear(total_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Initialize memory states
        self.reset_memory_states()
        
    def _get_stage_modulation(self, stage: DevelopmentalStage) -> Dict[str, float]:
        """Get stage-specific modulation factors"""
        base_factors = {
            DevelopmentalStage.NEWBORN: {
                'sensory': 0.4,
                'memory': 0.3,
                'emotional': 0.5,
                'decision': 0.3,
                'integration': 0.4
            },
            DevelopmentalStage.INFANT: {
                'sensory': 0.6,
                'memory': 0.5,
                'emotional': 0.7,
                'decision': 0.5,
                'integration': 0.6
            },
            DevelopmentalStage.EARLY_TODDLER: {
                'sensory': 0.8,
                'memory': 0.7,
                'emotional': 0.9,
                'decision': 0.7,
                'integration': 0.8
            }
        }
        return base_factors.get(stage, base_factors[DevelopmentalStage.EARLY_TODDLER])
        
    def reset_memory_states(self):
        """Reset LSTM and GRU memory states"""
        self.working_memory_state = None
        self.episodic_memory_state = None
        self.last_memory = None
        
    def process_sensory_input(self, visual_input: torch.Tensor, auditory_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process visual and auditory inputs"""
        visual_features = self.sensory_system['visual'](visual_input)
        auditory_features = self.sensory_system['auditory'](auditory_input)
        
        # Apply stage-specific modulation
        sensory_mod = self.modulation_factors['sensory']
        visual_features = visual_features * sensory_mod
        auditory_features = auditory_features * sensory_mod
        
        # Combine sensory features
        sensory_output = torch.cat([visual_features, auditory_features], dim=-1)
        return {'sensory_features': sensory_output}
        
    def process_memory(self, features: torch.Tensor, memory_type: str = 'all') -> Dict[str, torch.Tensor]:
        """Process information through memory systems"""
        # Process through working memory
        if memory_type in ['all', 'working']:
            working_output, self.working_memory_state = self.memory_system['working_memory'](
                features.unsqueeze(0),
                self.working_memory_state
            )
            working_output = working_output.squeeze(0)
        else:
            working_output = features
            
        # Process through episodic memory
        if memory_type in ['all', 'episodic']:
            episodic_output, self.episodic_memory_state = self.memory_system['episodic_memory'](
                features.unsqueeze(0),
                self.episodic_memory_state
            )
            episodic_output = episodic_output.squeeze(0)
        else:
            episodic_output = features
            
        # Process through semantic memory
        semantic_output = self.memory_system['semantic_memory'](features)
        
        # Integrate memory outputs
        memory_combined = torch.cat([
            working_output,
            episodic_output,
            semantic_output
        ], dim=-1)
        
        memory_output = self.memory_system['memory_integration'](memory_combined)
        
        # Apply memory smoothing
        if self.last_memory is not None:
            memory_output = (
                self.memory_alpha * self.last_memory + 
                (1 - self.memory_alpha) * memory_output
            )
        self.last_memory = memory_output.detach()
        
        # Apply stage-specific modulation
        memory_mod = self.modulation_factors['memory']
        memory_output = memory_output * memory_mod
        
        return {
            'working_memory': working_output,
            'episodic_memory': episodic_output,
            'semantic_memory': semantic_output,
            'integrated_memory': memory_output
        }
        
    def process_emotions(self, features: torch.Tensor, current_emotions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process information through emotional system"""
        # Generate new emotions
        emotion_input = torch.cat([features, current_emotions], dim=-1)
        new_emotions = torch.sigmoid(self.emotional_system['emotion_generator'](emotion_input))
        
        # Apply stage-specific modulation
        emotional_mod = self.modulation_factors['emotional']
        new_emotions = new_emotions * emotional_mod
        
        # Modulate features with emotions
        emotional_features = torch.cat([features, new_emotions], dim=-1)
        modulated_output = self.emotional_system['emotion_modulator'](emotional_features)
        
        # Update brain state
        joy, trust, fear, surprise = new_emotions.squeeze().tolist()
        self.brain_state.emotional_valence = (joy + trust - fear) / 2
        self.brain_state.arousal = (fear + surprise) / 2
        
        return {
            'emotions': new_emotions,
            'modulated_features': modulated_output
        }
        
    def make_decision(self, sensory_features: torch.Tensor, memory_features: torch.Tensor, 
                     emotional_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make decisions based on integrated information"""
        # Combine all features for decision making
        combined_features = torch.cat([
            sensory_features,
            memory_features,
            emotional_features
        ], dim=-1)
        
        # Apply stage-specific modulation
        decision_mod = self.modulation_factors['decision']
        
        # Select action
        action_features = self.decision_system['action_selector'](combined_features)
        action_features = action_features * decision_mod
        
        # Generate response
        response = self.decision_system['response_generator'](action_features)
        
        # Update brain state
        self.brain_state.attention = action_features.mean().item()
        
        return {
            'action': action_features,
            'response': response
        }
        
    def learn(self, features: torch.Tensor, reward: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update learning system based on reward"""
        # Ensure features have correct size
        if features.shape[-1] != self.hidden_dim:
            features = features[:, :self.hidden_dim]
            
        # Predict reward
        reward_prediction = self.learning_system['reinforcement'](features)
        
        # Calculate prediction error
        prediction_error = reward - reward_prediction
        
        # Adapt based on prediction error
        adapted_features = self.learning_system['adaptation'](features)
        
        return {
            'prediction_error': prediction_error,
            'adapted_features': adapted_features
        }
        
    def forward(self, visual_input: torch.Tensor, auditory_input: torch.Tensor, 
                current_emotions: torch.Tensor) -> Dict[str, Any]:
        """Process input through all brain systems"""
        # Process sensory input
        sensory_output = self.process_sensory_input(visual_input, auditory_input)
        
        # Process through memory systems
        memory_output = self.process_memory(sensory_output['sensory_features'])
        
        # Process through emotional system
        emotional_output = self.process_emotions(
            memory_output['integrated_memory'],
            current_emotions
        )
        
        # Make decision
        decision_output = self.make_decision(
            sensory_output['sensory_features'],
            memory_output['integrated_memory'],
            emotional_output['modulated_features']
        )
        
        # Integrate all outputs
        integrated_features = torch.cat([
            sensory_output['sensory_features'],  # hidden_dim
            memory_output['integrated_memory'],   # hidden_dim
            emotional_output['modulated_features'], # hidden_dim
            decision_output['action']  # hidden_dim // 2
        ], dim=-1)
        
        # Apply stage-specific modulation
        integration_mod = self.modulation_factors['integration']
        final_output = self.integration_network(integrated_features)
        
        # Apply tanh and scale to get values in desired range
        final_output = torch.tanh(final_output) * integration_mod
        final_output = (final_output + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Update brain state
        self.brain_state.consciousness = final_output.mean().item()
        self.brain_state.stress = max(0.1, min(0.9, 1 - self.brain_state.emotional_valence))
        self.brain_state.fatigue = min(1.0, self.brain_state.fatigue + 0.01)
        
        # Update neurotransmitters
        self.brain_state.neurotransmitters['dopamine'] = min(1.0, 0.5 + self.brain_state.emotional_valence * 0.3)
        self.brain_state.neurotransmitters['serotonin'] = min(1.0, 0.5 + self.brain_state.emotional_valence * 0.2)
        self.brain_state.neurotransmitters['norepinephrine'] = min(1.0, self.brain_state.arousal)
        self.brain_state.neurotransmitters['gaba'] = min(1.0, 1.0 - self.brain_state.arousal * 0.5)
        self.brain_state.neurotransmitters['glutamate'] = min(1.0, self.brain_state.arousal)
        
        return {
            'sensory': sensory_output,
            'memory': memory_output,
            'emotional': emotional_output,
            'decision': decision_output,
            'integrated': final_output
        }

    def get_brain_state(self) -> Dict[str, Any]:
        """Get the current brain state"""
        return {
            'arousal': self.brain_state.arousal,
            'attention': self.brain_state.attention,
            'emotional_valence': self.brain_state.emotional_valence,
            'consciousness': self.brain_state.consciousness,
            'stress': self.brain_state.stress,
            'fatigue': self.brain_state.fatigue,
            'neurotransmitters': dict(self.brain_state.neurotransmitters)
        }

class DigitalChild:
    def __init__(self, initial_stage: DevelopmentalStage = DevelopmentalStage.NEWBORN):
        # Initialize curriculum and development tracking
        self.curriculum = CurriculumManager()
        self.curriculum.current_stage = initial_stage
        
        # Initialize neural components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize integrated brain architecture
        self.brain = IntegratedBrain(input_dim=384, hidden_dim=256, stage=initial_stage).to(self.device)
        
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
        
        # Initialize emotional memory system
        self.emotional_memory_system = EmotionalMemorySystem(
            persist_dir="emotional_memories",
            sentence_transformer_model="all-MiniLM-L6-v2"
        )
        
        # Initialize integrated development tracking
        self.integrated_metrics = {
            'visual_cognitive_integration': 0.0,
            'emotional_response_strength': 0.0,
            'learning_rate': 0.0,
            'memory_formation': 0.0,
            'attention_control': 0.0
        }
        
        # Initialize brain state based on emotional state
        self._update_brain_state_from_emotional()
        
    def _update_brain_state_from_emotional(self):
        """Update brain state based on emotional state"""
        joy, trust, fear, surprise = self.emotional_state.cpu().tolist()
        
        # Calculate emotional valence (-1 to 1)
        valence = (joy + trust - fear - max(0, surprise - 0.5)) / 2
        
        # Calculate arousal (0 to 1)
        arousal = (surprise + fear + joy) / 3
        
        # Update brain state
        self.brain.brain_state.emotional_valence = valence
        self.brain.brain_state.arousal = arousal
        
        # Update neurotransmitters
        self.brain.brain_state.neurotransmitters['dopamine'] = min(1.0, 0.5 + valence * 0.3)
        self.brain.brain_state.neurotransmitters['serotonin'] = min(1.0, 0.5 + valence * 0.2)
        self.brain.brain_state.neurotransmitters['norepinephrine'] = min(1.0, arousal)
        self.brain.brain_state.neurotransmitters['gaba'] = min(1.0, 1.0 - arousal * 0.5)
        self.brain.brain_state.neurotransmitters['glutamate'] = min(1.0, arousal)
        
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
        self.curriculum.current_stage = stage
        self.curriculum.stage_progress = max(0.0, min(1.0, progress))
        
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
        current_stage_months = self.curriculum.stage_durations[self.curriculum.current_stage]
        total_months += int(current_stage_months * self.curriculum.stage_progress)
        
        return total_months

    def update_emotions(self, new_emotions: torch.Tensor):
        """Update emotional state with momentum and constraints"""
        # Apply momentum to smooth emotional transitions
        momentum = 0.7  # Higher value = more emotional inertia
        self.emotional_state = (momentum * self.emotional_state + 
                              (1 - momentum) * new_emotions)
        
        # Ensure values stay in valid range
        self.emotional_state = torch.clamp(self.emotional_state, 0.1, 1.0)
        
        # Update brain state based on new emotional state
        self._update_brain_state_from_emotional()
        
        return self.emotional_state

    def get_emotional_memories(self, query: str, limit: int = 3) -> List[Dict]:
        """Retrieve emotional memories related to the query"""
        try:
            # Get current emotional state
            emotional_state = {
                'joy': float(self.emotional_state[0]),
                'trust': float(self.emotional_state[1]),
                'fear': float(self.emotional_state[2]),
                'surprise': float(self.emotional_state[3])
            }
            
            # Get memories from emotional memory system
            memories = self.emotional_memory_system.retrieve_similar_memories(
                query=query,
                emotional_state=emotional_state,
                n_results=limit
            )
            
            return memories
            
        except Exception as e:
            print(f"Error retrieving emotional memories: {str(e)}")
            return []

    def express_feeling(self) -> str:
        """Express current emotional state in words"""
        joy, trust, fear, surprise = self.emotional_state.cpu().tolist()
        
        # Get stage-appropriate vocabulary
        if self.curriculum.current_stage in [DevelopmentalStage.NEWBORN, DevelopmentalStage.INFANT]:
            if fear > 0.7:
                return "crying"
            elif joy > 0.7:
                return "cooing happily"
            elif surprise > 0.7:
                return "startled"
            else:
                return "calm"
                
        elif self.curriculum.current_stage in [DevelopmentalStage.EARLY_TODDLER, DevelopmentalStage.LATE_TODDLER]:
            if fear > 0.7:
                return "scared"
            elif joy > 0.7:
                return "happy"
            elif trust > 0.7:
                return "good"
            elif surprise > 0.7:
                return "wow"
            else:
                return "okay"
                
        else:  # More advanced stages
            # Calculate primary emotion
            emotions = {
                'joy': joy,
                'trust': trust,
                'fear': fear,
                'surprise': surprise
            }
            primary_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Express with more nuanced vocabulary
            if primary_emotion[1] > 0.7:  # Strong emotion
                if primary_emotion[0] == 'joy':
                    return "very happy and excited"
                elif primary_emotion[0] == 'trust':
                    return "feeling safe and content"
                elif primary_emotion[0] == 'fear':
                    return "feeling anxious and worried"
                else:  # surprise
                    return "amazed and curious"
            elif primary_emotion[1] > 0.4:  # Moderate emotion
                if primary_emotion[0] == 'joy':
                    return "cheerful"
                elif primary_emotion[0] == 'trust':
                    return "comfortable"
                elif primary_emotion[0] == 'fear':
                    return "uneasy"
                else:  # surprise
                    return "interested"
            else:  # Mild or mixed emotions
                return "feeling balanced"