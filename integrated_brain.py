"""
Integrated Brain Module for Neural Child Development
Created by: Christopher Celaya

This module implements the integrated brain architecture that combines multiple neural components
and incorporates Q-Learning for decision making.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from enum import Enum, auto
from datetime import datetime
from q_learning import QLearningSystem
from language_development import LanguageDevelopment
from heartbeat_system import HeartbeatSystem
import chromadb
from emotional_memory_system import EmotionalMemoryEntry

class BrainState:
    """Class to track brain state"""
    def __init__(self, sensory_state: torch.Tensor = None, memory_state: torch.Tensor = None, emotional_state: torch.Tensor = None, attention_state: torch.Tensor = None, consciousness_state: torch.Tensor = None):
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
        if sensory_state is not None:
            self.sensory_state = sensory_state
        if memory_state is not None:
            self.memory_state = memory_state
        if emotional_state is not None:
            self.emotional_state = emotional_state
        if attention_state is not None:
            self.attention_state = attention_state
        if consciousness_state is not None:
            self.consciousness_state = consciousness_state
            
    def to_tensor(self) -> torch.Tensor:
        """Convert brain state to tensor for Q-Learning"""
        # Create list of state components
        state_components = []
        
        # Add sensory state if available
        if hasattr(self, 'sensory_state'):
            state_components.append(self.sensory_state.flatten())
            
        # Add memory state if available
        if hasattr(self, 'memory_state'):
            state_components.append(self.memory_state.flatten())
            
        # Add emotional state if available
        if hasattr(self, 'emotional_state'):
            state_components.append(self.emotional_state.flatten())
            
        # Add attention state if available
        if hasattr(self, 'attention_state'):
            state_components.append(self.attention_state.flatten())
            
        # Add consciousness state if available
        if hasattr(self, 'consciousness_state'):
            state_components.append(self.consciousness_state.flatten())
            
        # Add scalar values
        scalar_values = torch.tensor([
            self.emotional_valence,
            self.arousal,
            self.attention,
            self.consciousness,
            self.stress,
            self.fatigue
        ])
        state_components.append(scalar_values)
        
        # Add neurotransmitter values
        neurotransmitter_values = torch.tensor([
            self.neurotransmitters['dopamine'],
            self.neurotransmitters['serotonin'],
            self.neurotransmitters['norepinephrine'],
            self.neurotransmitters['gaba'],
            self.neurotransmitters['glutamate']
        ])
        state_components.append(neurotransmitter_values)
        
        # Concatenate all components
        return torch.cat(state_components)

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

class IntegratedBrain(nn.Module):
    """Integrated brain architecture combining multiple neural components"""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        """Initialize integrated brain architecture"""
        super().__init__()
        
        # Initialize dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = hidden_dim * 2
        self.action_dim = 32  # Number of possible actions
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize brain state
        self.brain_state = BrainState()
        
        # Initialize developmental stage
        self.stage = DevelopmentalStage.NEWBORN
        self.stage_progress = 0.0
        
        # Initialize language development
        self.language_development = LanguageDevelopment()
        
        # Initialize heartbeat system
        self.heartbeat = HeartbeatSystem()
        
        # Initialize neural components
        self.sensory_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        ).to(self.device)
        
        # Initialize memory network
        self.memory_network = nn.Sequential(
            nn.Linear(self.action_dim, 64),  # First layer matches action_dim input
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),  # Output matches memory state size
            nn.ReLU(),
            nn.LayerNorm(32)
        ).to(self.device)
        
        self.emotional_network = nn.Sequential(
            nn.Linear(32, 64),  # Match memory network output
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Tanh()
        ).to(self.device)
        
        # Multi-head attention for attention network
        self.attention = nn.MultiheadAttention(
            embed_dim=32,  # Match memory state size
            num_heads=4,
            batch_first=True
        ).to(self.device)
        
        self.consciousness_network = nn.Sequential(
            nn.Linear(32, 64),  # Match memory state size
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 8),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize Q-Learning system
        self.q_learning = QLearningSystem(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            device=self.device
        )
        
        # Initialize memory states with correct dimensions
        self.memory_states = {
            'working': torch.zeros(32, device=self.device),
            'short_term': torch.zeros(32, device=self.device),
            'long_term': torch.zeros(32, device=self.device)
        }
        
        # Initialize metrics tracking
        self.metrics = {
            'q_learning_rewards': [],
            'q_learning_losses': [],
            'training_loss': 0.0
        }
        
        # Register get_training_loss as a method
        self.get_training_loss = lambda: self.metrics['training_loss']
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client()
        
        # Try to get existing collections or create new ones
        try:
            self.emotional_collection = self.chroma_client.get_collection(name="emotional_memories")
            print("Using existing emotional memories collection")
        except chromadb.errors.InvalidCollectionException:
            self.emotional_collection = self.chroma_client.create_collection(
                name="emotional_memories",
                metadata={"hnsw:space": "cosine"}
            )
            print("Created new emotional memories collection")
            
        try:
            self.episodic_collection = self.chroma_client.get_collection(name="episodic_memories")
            print("Using existing episodic memories collection")
        except chromadb.errors.InvalidCollectionException:
            self.episodic_collection = self.chroma_client.create_collection(
                name="episodic_memories",
                metadata={"hnsw:space": "cosine"}
            )
            print("Created new episodic memories collection")
            
        try:
            self.semantic_collection = self.chroma_client.get_collection(name="semantic_memories")
            print("Using existing semantic memories collection")
        except chromadb.errors.InvalidCollectionException:
            self.semantic_collection = self.chroma_client.create_collection(
                name="semantic_memories",
                metadata={"hnsw:space": "cosine"}
            )
            print("Created new semantic memories collection")
            
        # Set collection attributes for compatibility
        self.emotional_memory_collection = self.emotional_collection
        self.episodic_memory_collection = self.episodic_collection
        self.semantic_memory_collection = self.semantic_collection
    
    def get_state(self) -> BrainState:
        """Get current brain state"""
        # Create a basic brain state
        state = BrainState()
        
        # Add memory states
        state.memory_state = torch.cat([
            self.memory_states['working'],
            self.memory_states['short_term'],
            self.memory_states['long_term']
        ]).unsqueeze(0)
        
        # Add dummy sensory state
        state.sensory_state = torch.zeros(1, self.hidden_dim, device=self.device)
        
        # Add dummy emotional state
        state.emotional_state = torch.zeros(1, 32, device=self.device)
        
        # Add dummy attention state
        state.attention_state = torch.zeros(1, 32, device=self.device)
        
        # Add dummy consciousness state
        state.consciousness_state = torch.zeros(1, 8, device=self.device)
        
        return state
    
    def make_decision(self, state: BrainState) -> int:
        """Make decision based on current state using Q-Learning"""
        # Convert BrainState to tensor for Q-Learning
        state_tensor = state.to_tensor().to(self.device)
        
        # Get action from Q-Learning system
        action = self.q_learning.select_action(state_tensor)
        
        return action  # Return the action index directly
    
    def step_environment(self, action: int) -> BrainState:
        """Execute action and return new state"""
        # Convert action to one-hot encoding and reshape for batch processing
        action_encoding = torch.zeros(1, self.action_dim, device=self.device)  # Add batch dimension
        action_encoding[0, action] = 1.0
        
        # Update memory states based on action
        memory_update = self.memory_network(action_encoding)  # Now expects [batch_size, action_dim]
        
        # Update working memory with new information
        self.memory_states['working'] = (
            0.7 * self.memory_states['working'] + 
            0.3 * memory_update.squeeze()  # Remove batch dimension
        )
        
        # Create new brain state
        new_state = BrainState()
        
        # Update brain state values based on memory update
        new_state.emotional_valence = torch.mean(memory_update).item()
        new_state.arousal = torch.std(memory_update).item()
        new_state.attention = torch.max(memory_update).item()
        new_state.consciousness = torch.median(memory_update).item()
        
        # Add memory states
        new_state.memory_state = torch.cat([
            self.memory_states['working'],
            self.memory_states['short_term'],
            self.memory_states['long_term']
        ]).unsqueeze(0)
        
        # Add sensory state
        new_state.sensory_state = torch.zeros(1, self.hidden_dim, device=self.device)
        
        # Add emotional state
        new_state.emotional_state = torch.zeros(1, 32, device=self.device)
        
        # Add attention state
        new_state.attention_state = torch.zeros(1, 32, device=self.device)
        
        # Add consciousness state
        new_state.consciousness_state = torch.zeros(1, 8, device=self.device)
        
        return new_state
    
    def train(self) -> None:
        """Train the integrated brain architecture"""
        # Train Q-Learning system
        loss = self.q_learning.train_step()
        if loss is not None:
            self.metrics['q_learning_losses'].append(loss)
            self.metrics['training_loss'] = loss
    
    def save_brain_state(self, path: str) -> None:
        """Save brain state to file"""
        state_dict = {
            'sensory_network': self.sensory_network.state_dict(),
            'memory_network': self.memory_network.state_dict(),
            'emotional_network': self.emotional_network.state_dict(),
            'attention': self.attention.state_dict(),
            'consciousness_network': self.consciousness_network.state_dict(),
            'q_learning': self.q_learning.state_dict(),
            'metrics': self.metrics,
            'memory_states': {k: v.cpu() for k, v in self.memory_states.items()},
            'developmental_stage': self.stage.value,
            'stage_progress': self.stage_progress
        }
        torch.save(state_dict, path)
    
    def load_brain_state(self, path: str) -> None:
        """Load brain state from file"""
        state_dict = torch.load(path)
        self.sensory_network.load_state_dict(state_dict['sensory_network'])
        self.memory_network.load_state_dict(state_dict['memory_network'])
        self.emotional_network.load_state_dict(state_dict['emotional_network'])
        self.attention.load_state_dict(state_dict['attention'])
        self.consciousness_network.load_state_dict(state_dict['consciousness_network'])
        self.q_learning.load_state_dict(state_dict['q_learning'])
        self.metrics = state_dict['metrics']
        self.memory_states = {k: v.to(self.device) for k, v in state_dict['memory_states'].items()}
        self.stage = DevelopmentalStage(state_dict['developmental_stage'])
        self.stage_progress = state_dict['stage_progress']
    
    def get_reward(self) -> float:
        """Calculate reward based on learning progress and developmental stage"""
        # Get learning signal from memory network activity
        memory_activity = torch.mean(self.memory_states['working']).item()
        
        # Calculate base reward components
        learning_progress = memory_activity * 0.4
        emotional_stability = self.brain_state.emotional_valence * 0.2
        attention_focus = self.brain_state.attention * 0.2
        consciousness_level = self.brain_state.consciousness * 0.2
        
        # Calculate stage-specific bonus
        stage_bonus = {
            DevelopmentalStage.NEWBORN: 0.1,
            DevelopmentalStage.INFANT: 0.15,
            DevelopmentalStage.EARLY_TODDLER: 0.2,
            DevelopmentalStage.LATE_TODDLER: 0.25,
            DevelopmentalStage.EARLY_PRESCHOOL: 0.3,
            DevelopmentalStage.LATE_PRESCHOOL: 0.35,
            DevelopmentalStage.EARLY_CHILDHOOD: 0.4,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.45,
            DevelopmentalStage.LATE_CHILDHOOD: 0.5,
            DevelopmentalStage.PRE_ADOLESCENT: 0.55,
            DevelopmentalStage.EARLY_TEEN: 0.6,
            DevelopmentalStage.MID_TEEN: 0.65,
            DevelopmentalStage.LATE_TEEN: 0.7,
            DevelopmentalStage.YOUNG_ADULT: 0.75,
            DevelopmentalStage.EARLY_TWENTIES: 0.8
        }.get(self.stage, 0.1)
        
        # Calculate total reward
        base_reward = (
            learning_progress +
            emotional_stability +
            attention_focus +
            consciousness_level
        ) / 4.0
        
        # Apply stage bonus and progress multiplier
        reward = base_reward * (1 + stage_bonus) * (1 + self.stage_progress)
        
        # Update metrics
        self.metrics['training_loss'] = 1.0 - reward
        
        return max(0.0, min(1.0, reward))

    def get_brain_state(self) -> Dict[str, Any]:
        """Get the current state of the brain including memory, emotion, and development metrics"""
        heartbeat_info = self.heartbeat.get_current_heartbeat()
        
        return {
            'memory_state': self.memory_states['working'].tolist(),
            'emotion_state': {
                'emotional_valence': self.brain_state.emotional_valence,
                'arousal': self.brain_state.arousal,
                'attention': self.brain_state.attention,
                'consciousness': self.brain_state.consciousness,
                'stress': self.brain_state.stress,
                'fatigue': self.brain_state.fatigue
            },
            'development_stage': self.stage.name,
            'language_metrics': self.language_development.get_metrics(),
            'trust_level': 0.5,  # Placeholder for trust level
            'attention_focus': self.brain_state.attention,
            'consciousness_level': self.brain_state.consciousness,
            'metrics': dict(self.metrics),
            'heartbeat': heartbeat_info
        }

    def process_emotions(self, features: torch.Tensor, current_emotions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process information through emotional system.
        
        Args:
            features (torch.Tensor): Input features
            current_emotions (torch.Tensor): Current emotional state
            
        Returns:
            Dict containing emotional processing results
        """
        # Generate new emotions
        emotion_input = torch.cat([features, current_emotions], dim=-1)
        new_emotions = torch.sigmoid(self.emotional_network(emotion_input))
        
        # Update brain state
        self.brain_state.emotional_valence = new_emotions[0][0].item()
        self.brain_state.arousal = new_emotions[0][1].item()
        self.brain_state.attention = new_emotions[0][2].item()
        self.brain_state.consciousness = new_emotions[0][3].item()
        
        # Update heartbeat based on emotional state
        heartbeat_update = self.heartbeat.update_from_emotional_state({
            'joy': max(0, self.brain_state.emotional_valence),
            'trust': max(0, self.brain_state.emotional_valence * 0.8),
            'fear': max(0, -self.brain_state.emotional_valence) * self.brain_state.arousal,
            'surprise': self.brain_state.arousal * 0.5,
            'attention': self.brain_state.attention,
            'anger': max(0, -self.brain_state.emotional_valence) * self.brain_state.arousal * 0.7,
            'anticipation': self.brain_state.attention * 0.6
        })
        
        return {
            'emotions': new_emotions,
            'modulated_features': features * new_emotions.mean(),
            'heartbeat': heartbeat_update
        }

    def process_memory(self, features: torch.Tensor, memory_type: str = 'all') -> Dict[str, torch.Tensor]:
        """Process information through memory systems.
        
        Args:
            features (torch.Tensor): Input features
            memory_type (str): Type of memory to process ('all', 'working', 'short_term', 'long_term')
            
        Returns:
            Dict containing memory processing results
        """
        # Process through working memory
        if memory_type in ['all', 'working']:
            working_output = self.memory_network(features)
            self.memory_states['working'] = working_output.squeeze()
        else:
            working_output = features
            
        # Process through emotional memory
        emotional_output = self.emotional_network(working_output)
        
        # Update brain state
        self.brain_state.emotional_valence = emotional_output[0][0].item()
        self.brain_state.arousal = emotional_output[0][1].item()
        self.brain_state.attention = emotional_output[0][2].item()
        self.brain_state.consciousness = emotional_output[0][3].item()
        
        # Update heartbeat based on memory processing
        heartbeat_update = self.heartbeat.process_memory_trigger(
            memory_valence=self.brain_state.emotional_valence,
            memory_intensity=self.brain_state.arousal
        )
        
        # Memory consolidation
        if memory_type == 'all':
            # Update short-term memory
            self.memory_states['short_term'] = (
                0.8 * self.memory_states['short_term'] +
                0.2 * working_output.squeeze()
            )
            
            # Update long-term memory with important information
            if self.brain_state.arousal > 0.7 or abs(self.brain_state.emotional_valence) > 0.6:
                self.memory_states['long_term'] = (
                    0.95 * self.memory_states['long_term'] +
                    0.05 * working_output.squeeze()
                )
        
        return {
            'working_memory': working_output,
            'emotional_memory': emotional_output,
            'heartbeat': heartbeat_update,
            'memory_states': {k: v.clone() for k, v in self.memory_states.items()}
        }

    def process_attention(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process information through attention system.
        
        Args:
            features (torch.Tensor): Input features
            
        Returns:
            Dict containing attention processing results
        """
        # Ensure features have batch and sequence dimensions
        if len(features.shape) == 2:
            features = features.unsqueeze(0)  # Add batch dimension
            
        # Apply attention mechanism
        attended_features, attention_weights = self.attention(
            features, features, features
        )
        
        # Update brain state
        self.brain_state.attention = attention_weights.mean().item()
        
        return {
            'attended_features': attended_features,
            'attention_weights': attention_weights
        }

    def process_consciousness(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process information through consciousness system.
        
        Args:
            features (torch.Tensor): Input features
            
        Returns:
            Dict containing consciousness processing results
        """
        # Process through consciousness network
        consciousness_output = self.consciousness_network(features)
        
        # Update brain state
        self.brain_state.consciousness = consciousness_output.mean().item()
        
        # Update neurotransmitter levels based on consciousness
        self.brain_state.neurotransmitters['dopamine'] = min(1.0, 0.5 + consciousness_output[0][0].item() * 0.3)
        self.brain_state.neurotransmitters['serotonin'] = min(1.0, 0.5 + consciousness_output[0][1].item() * 0.2)
        self.brain_state.neurotransmitters['norepinephrine'] = min(1.0, consciousness_output[0][2].item())
        self.brain_state.neurotransmitters['gaba'] = min(1.0, 1.0 - consciousness_output[0][3].item() * 0.5)
        self.brain_state.neurotransmitters['glutamate'] = min(1.0, consciousness_output[0][4].item())
        
        return {
            'consciousness_level': consciousness_output,
            'neurotransmitter_levels': dict(self.brain_state.neurotransmitters)
        }

    def forward(self, visual_input: torch.Tensor, auditory_input: torch.Tensor, 
                current_emotions: torch.Tensor) -> Dict[str, Any]:
        """Process input through all brain systems.
        
        Args:
            visual_input (torch.Tensor): Visual input features
            auditory_input (torch.Tensor): Auditory input features
            current_emotions (torch.Tensor): Current emotional state
            
        Returns:
            Dict containing results from all processing systems
        """
        # Process sensory input
        sensory_features = self.sensory_network(torch.cat([visual_input, auditory_input], dim=-1))
        
        # Process through attention system
        attention_output = self.process_attention(sensory_features)
        attended_features = attention_output['attended_features']
        
        # Process through memory systems
        memory_output = self.process_memory(attended_features)
        
        # Process through emotional system
        emotional_output = self.process_emotions(
            memory_output['working_memory'],
            current_emotions
        )
        
        # Process through consciousness system
        consciousness_output = self.process_consciousness(emotional_output['modulated_features'])
        
        # Integrate all outputs
        integrated_features = torch.cat([
            attended_features.squeeze(),
            memory_output['working_memory'].squeeze(),
            emotional_output['emotions'].squeeze(),
            consciousness_output['consciousness_level'].squeeze()
        ], dim=-1)
        
        # Update brain state
        self.brain_state.consciousness = consciousness_output['consciousness_level'].mean().item()
        self.brain_state.stress = max(0.1, min(0.9, 1 - self.brain_state.emotional_valence))
        self.brain_state.fatigue = min(1.0, self.brain_state.fatigue + 0.01)
        
        return {
            'attention': attention_output,
            'memory': memory_output,
            'emotional': emotional_output,
            'consciousness': consciousness_output,
            'integrated': integrated_features
        } 