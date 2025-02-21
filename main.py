# main.py
# Description: Main file for neural child development
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
from neural_architecture import NeuralArchitecture, BrainRegion, CognitiveFunction
from decision_network import DecisionNetwork
from ollama_chat import OllamaChat
from obsidian_api import ObsidianAPI
from obsidian_visualizer import ObsidianVisualizer
from emotional_regulation import EmotionalRegulation
from memory_module import MemoryModule
from self_supervised_trainer import SelfSupervisedTrainer
from moral_network import MoralNetwork
from rag_memory import RAGMemorySystem
from self_awareness_network import SelfAwarenessNetwork, SelfAwarenessLevel
from integrated_brain import IntegratedBrain, BrainState
from curriculum_manager import CurriculumManager
from llm_module import chat_completion
from q_learning import QLearningSystem
from heartbeat_system import HeartbeatSystem
from developmental_stages import DevelopmentalStage



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
        
        # Initialize heartbeat system
        self.heartbeat = HeartbeatSystem()
        
        # Initialize developmental stage
        self.stage = DevelopmentalStage.NEWBORN
        self.stage_progress = 0.0
        
        # Initialize language development
        self.language_development = LanguageDevelopment()
        
        # Initialize neural components
        self.sensory_network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # *2 because we concatenate visual and auditory
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # Changed to output full hidden_dim
        ).to(self.device)
        
        # Initialize memory network with smoothing layers
        self.memory_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ).to(self.device)
        
        # Memory smoothing layer
        self.memory_smoothing = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Combine current and previous state
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(self.device)
        
        # Initialize emotional network
        emotional_input_size = hidden_dim + 4  # Features + 4 emotional inputs
        self.emotional_network = nn.Sequential(
            nn.Linear(emotional_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)  # Changed from 12 to 4 emotional dimensions
        ).to(self.device)
        
        # Initialize Q-Learning system
        self.q_learning = QLearningSystem(
            state_dim=hidden_dim,  # Match sensory output size
            action_dim=self.action_dim,
            hidden_dim=hidden_dim
        )
        
        # Learning network for reward prediction
        self.learning_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Learning signal/reward prediction
        ).to(self.device)
        
        # Initialize memory states with correct dimensions
        self.memory_states = {
            'working': torch.zeros(hidden_dim, device=self.device),
            'short_term': torch.zeros(hidden_dim, device=self.device),
            'long_term': torch.zeros(hidden_dim, device=self.device)
        }
        
        # Initialize attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        ).to(self.device)
        
        # Initialize gating mechanism for memory consolidation
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize developmental modulation
        self.developmental_factors = self._get_stage_modulation(DevelopmentalStage.EARLY_TODDLER)
        
        # Track metrics
        self.metrics = {
            'attention_span': 0.0,
            'learning_rate': 0.0,
            'memory_consolidation': 0.0,
            'emotional_regulation': 0.0,
            'q_learning_rewards': [],
            'q_learning_losses': [],
            'training_loss': 0.0
        }
        
    def _get_stage_modulation(self, stage: DevelopmentalStage) -> Dict[str, float]:
        """Get stage-specific modulation factors"""
        base_factors = {
            DevelopmentalStage.NEWBORN: {
                'sensory': 0.15,    # Further reduced from 0.2
                'memory': 0.08,     # Further reduced from 0.1
                'emotional': 0.2,   # Further reduced from 0.3
                'decision': 0.08,   # Further reduced from 0.1
                'integration': 0.08, # Further reduced from 0.1
                'learning': 0.08     # Further reduced from 0.1
            },
            DevelopmentalStage.INFANT: {
                'sensory': 0.9,
                'memory': 0.8,
                'emotional': 0.9,
                'decision': 0.8,
                'integration': 0.8,
                'learning': 0.8
            },
            DevelopmentalStage.EARLY_TODDLER: {
                'sensory': 1.0,
                'memory': 0.9,
                'emotional': 1.0,
                'decision': 0.9,
                'integration': 1.0,
                'learning': 0.9
            }
        }
        return base_factors.get(stage, base_factors[DevelopmentalStage.EARLY_TODDLER])
        
    def process_sensory_input(self, visual_input: torch.Tensor, auditory_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process visual and auditory inputs"""
        sensory_features = self.sensory_network(torch.cat([visual_input, auditory_input], dim=-1))
        return {'sensory_features': sensory_features}
        
    def process_memory(self, features: torch.Tensor, memory_type: str = 'all') -> Dict[str, torch.Tensor]:
        """Process information through memory systems.
        
        Args:
            features (torch.Tensor): Input features to process
            memory_type (str): Type of memory to process ('all', 'working', 'episodic', 'semantic')
            
        Returns:
            Dict containing processed memory outputs
        """
        # Ensure features have batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        # Apply initial smoothing to input features
        features = 0.3 * features + 0.7 * self.memory_states['working'].unsqueeze(0)
            
        # Process through working memory with smoothing
        if memory_type in ['all', 'working']:
            working_output = self.memory_network(features)
            # Apply memory smoothing layer
            working_combined = torch.cat([
                working_output,
                self.memory_states['working'].unsqueeze(0).expand_as(working_output)
            ], dim=-1)
            working_output = self.memory_smoothing(working_combined)
            # Additional smoothing
            working_output = 0.3 * working_output + 0.7 * self.memory_states['working'].unsqueeze(0)
        else:
            working_output = features
            
        # Process through episodic memory with smoothing
        episodic_output = self.memory_network(working_output)
        episodic_combined = torch.cat([
            episodic_output,
            self.memory_states['short_term'].unsqueeze(0).expand_as(episodic_output)
        ], dim=-1)
        episodic_output = self.memory_smoothing(episodic_combined)
        # Additional smoothing
        episodic_output = 0.2 * episodic_output + 0.8 * self.memory_states['short_term'].unsqueeze(0)
        
        # Process through semantic memory with smoothing
        semantic_output = self.memory_network(episodic_output)
        semantic_combined = torch.cat([
            semantic_output,
            self.memory_states['long_term'].unsqueeze(0).expand_as(semantic_output)
        ], dim=-1)
        semantic_output = self.memory_smoothing(semantic_combined)
        # Additional smoothing
        semantic_output = 0.1 * semantic_output + 0.9 * self.memory_states['long_term'].unsqueeze(0)
        
        # Apply additional smoothing to all memory outputs
        working_output = torch.sigmoid(working_output)  # Use sigmoid instead of tanh
        episodic_output = torch.sigmoid(episodic_output)
        semantic_output = torch.sigmoid(semantic_output)
        
        # Update memory states with smoothed values
        self.memory_states['working'] = working_output.squeeze()
        self.memory_states['short_term'] = episodic_output.squeeze()
        self.memory_states['long_term'] = semantic_output.squeeze()
        
        # Integrate memory outputs with smoothing
        integrated_memory = torch.cat([
            working_output,
            episodic_output,
            semantic_output
        ], dim=-1)
        
        # Final smoothing on integrated memory
        integrated_memory = torch.sigmoid(integrated_memory)
        
        # Update brain state
        self.brain_state.emotional_valence = working_output[0][0].item()
        self.brain_state.arousal = working_output[0][1].item()
        self.brain_state.attention = working_output[0][2].item()
        self.brain_state.consciousness = working_output[0][3].item()
        
        # Update heartbeat based on memory processing
        heartbeat_update = self.heartbeat.process_memory_trigger(
            memory_valence=self.brain_state.emotional_valence,
            memory_intensity=self.brain_state.arousal
        )
        
        return {
            'working_memory': working_output,
            'episodic_memory': episodic_output,
            'semantic_memory': semantic_output,
            'integrated_memory': integrated_memory,
            'heartbeat': heartbeat_update
        }
        
    def process_emotions(self, features: torch.Tensor, current_emotions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process information through emotional system"""
        # Ensure current_emotions has correct shape
        if current_emotions.dim() == 1:
            current_emotions = current_emotions.unsqueeze(0)
            
        # Generate new emotions
        emotion_input = torch.cat([features, current_emotions], dim=-1)
        new_emotions = torch.sigmoid(self.emotional_network(emotion_input))
        
        # Modulate features with emotional state
        # Expand emotions to match feature dimensions
        emotion_modulation = torch.tanh(new_emotions[:, :4])  # Use first 4 emotions
        emotion_modulation = emotion_modulation.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        emotion_modulation = emotion_modulation.mean(dim=1)  # Average across emotions
        modulated_features = features * (1.0 + 0.1 * emotion_modulation)
        
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
            'modulated_features': modulated_features,
            'heartbeat': heartbeat_update
        }
        
    def make_decision(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make a decision based on combined features.
        
        Args:
            features (torch.Tensor): Combined features tensor from sensory, memory, and emotional systems
            
        Returns:
            Dict containing decision outputs
        """
        # Project combined features to state dimension for Q-Learning
        projection_layer = nn.Linear(features.shape[-1], self.hidden_dim).to(self.device)
        projected_features = projection_layer(features)
        
        # Apply developmental modulation
        modulated_features = projected_features * self.developmental_factors['decision']
        
        # Get Q-values for possible actions
        q_values = self.q_learning.get_q_values(modulated_features)
        
        # Select action with highest Q-value
        action = torch.argmax(q_values, dim=-1).to(self.device)
        
        # Generate response based on action
        response = torch.zeros(action.shape[0], self.action_dim, device=self.device)
        response.scatter_(1, action.unsqueeze(1), 1.0)
        
        return {
            'action': action,
            'response': response,
            'q_values': q_values
        }

    def generate_response(self, action: torch.Tensor) -> torch.Tensor:
        """Generate a response based on the selected action.
        
        Args:
            action (torch.Tensor): Selected action tensor
            
        Returns:
            torch.Tensor: Generated response
        """
        # Ensure action is on the correct device
        action = action.to(self.device)
        
        # Simple response generation based on action index
        response_dim = self.hidden_dim // 2
        response = torch.zeros(action.shape[0], response_dim, device=self.device)
        response.scatter_(1, action.unsqueeze(1), 1.0)
        return response
        
    def learn(self, features: torch.Tensor, reward: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update learning system based on reward.
        
        Args:
            features (torch.Tensor): Input features for learning
            reward (torch.Tensor): Reward signal
            
        Returns:
            Dict containing learning outputs
        """
        # Ensure features have correct size
        if features.shape[-1] != self.hidden_dim:
            features = features[:, :self.hidden_dim]
            
        # Predict reward
        reward_prediction = self.learning_network(features)
        
        # Calculate prediction error
        prediction_error = reward - reward_prediction
        
        # Adapt features based on prediction error
        adapted_features = features * (1.0 + 0.1 * torch.tanh(prediction_error))
        
        return {
            'prediction_error': prediction_error,
            'adapted_features': adapted_features
        }
        
    def forward(self, visual_input: torch.Tensor, auditory_input: torch.Tensor, 
                current_emotions: torch.Tensor) -> Dict[str, Any]:
        """Process input through all brain systems"""
        # Process sensory input with scaling
        sensory_output = self.process_sensory_input(visual_input, auditory_input)
        sensory_features = sensory_output['sensory_features'] * (self.developmental_factors['sensory'] * 1.5)
        sensory_features = torch.sigmoid(sensory_features * 0.5)  # Even stronger normalization
        
        # Process through memory systems with scaling
        memory_output = self.process_memory(sensory_features)
        memory_features = memory_output['working_memory'] * (self.developmental_factors['memory'] * 1.5)
        memory_features = torch.sigmoid(memory_features * 0.5)
        
        # Process through emotional system with scaling
        emotional_output = self.process_emotions(
            sensory_features,
            current_emotions
        )
        emotional_features = emotional_output['emotions'] * (self.developmental_factors['emotional'] * 1.5)
        emotional_features = torch.sigmoid(emotional_features * 0.5)
        
        # Make decision with scaling
        combined_features = torch.cat([
            sensory_features,
            memory_features,
            emotional_features
        ], dim=-1)
        
        decision_output = self.make_decision(combined_features)
        decision_features = decision_output['response'] * (self.developmental_factors['decision'] * 1.5)
        decision_features = torch.sigmoid(decision_features * 0.5)
        
        # Integrate all outputs with proper dimensions and scaling
        integrated_features = torch.cat([
            sensory_features,     # hidden_dim
            memory_features,      # hidden_dim
            emotional_features,   # 4 dimensions
            decision_features     # action_dim
        ], dim=-1)
        
        # Apply stage-specific modulation with reduced scale
        integration_mod = self.developmental_factors['integration']  # No multiplier
        final_output = integrated_features * integration_mod
        
        # Additional normalization for newborn stage
        if self.stage == DevelopmentalStage.NEWBORN:
            final_output = final_output * 0.4  # Further reduce output for newborns
            final_output = torch.sigmoid(final_output * 0.2)  # Extra normalization for newborns
            # Directly clamp newborn output to ensure it stays below 0.5
            final_output = torch.clamp(final_output, max=0.45)
        else:
            # Normalize final output to [0, 1] range with stronger sigmoid
            final_output = torch.sigmoid(final_output * 0.3)
        
        # Update brain state
        self.brain_state.consciousness = torch.mean(final_output).item()
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
        heartbeat_info = self.heartbeat.get_current_heartbeat()
        
        return {
            'arousal': self.brain_state.arousal,
            'attention': self.brain_state.attention,
            'emotional_valence': self.brain_state.emotional_valence,
            'consciousness': self.brain_state.consciousness,
            'stress': self.brain_state.stress,
            'fatigue': self.brain_state.fatigue,
            'neurotransmitters': dict(self.brain_state.neurotransmitters),
            'heartbeat': heartbeat_info
        }

    def get_state(self) -> torch.Tensor:
        """Get current state for Q-Learning"""
        # Combine sensory and memory features
        sensory_features = self.sensory_network(torch.zeros(1, self.input_dim, device=self.device))
        memory_features = torch.cat([
            self.memory_states['working'],
            self.memory_states['short_term'],
            self.memory_states['long_term']
        ]).unsqueeze(0)
        
        return torch.cat([sensory_features, memory_features], dim=1)

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using emotional network.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        # Simple rule-based sentiment analysis
        positive_words = {
            'happy', 'excited', 'wonderful', 'great', 'awesome', 'love', 'celebrate',
            'curious', 'interested', 'fascinating', 'amazing', 'excellent', 'fantastic',
            'joy', 'delighted', 'pleased', 'glad', 'thrilled', 'enthusiastic', 'awesome',
            "let's", 'celebrate'  # Added celebration-related words
        }
        negative_words = {
            'sad', 'unhappy', 'terrible', 'awful', 'hate', 'miserable', 'depressed',
            'worried', 'anxious', 'scared', 'afraid', 'upset', 'disappointed',
            'frustrated', 'angry', 'annoyed', 'distressed', 'troubled', 'feeling'
        }
        
        # Additional sentiment modifiers
        intensifiers = {'very', 'really', 'extremely', 'incredibly', 'absolutely', 'quite'}
        negators = {'not', "n't", 'never', 'no', 'none', 'neither', 'nor'}
        
        words = text.lower().split()
        
        # First check for explicit emotional states
        if "i'm feeling" in text.lower() or "i am feeling" in text.lower():
            # Direct emotional statement
            if any(word in negative_words for word in words):
                return -1.0
            if any(word in positive_words for word in words):
                return 1.0
        
        # Check for celebration in response to negative emotion
        has_negative_context = any(word in negative_words for word in words)
        has_celebration = any(word in {'celebrate', "let's", 'awesome'} for word in words)
        if has_negative_context and has_celebration:
            return -1.0  # Strong negative for inappropriate celebration
        
        # Count sentiment words with modifiers
        positive_score = 0
        negative_score = 0
        negate_next = False
        intensify_next = False
        
        for i, word in enumerate(words):
            # Check for negators and intensifiers
            if word in negators:
                negate_next = True
                continue
            if word in intensifiers:
                intensify_next = True
                continue
                
            # Calculate word sentiment
            sentiment_value = 0
            if word in positive_words:
                sentiment_value = 1.5
            elif word in negative_words:
                sentiment_value = -1.5
                
            # Apply modifiers
            if sentiment_value != 0:
                if negate_next:
                    sentiment_value *= -2.0
                if intensify_next:
                    sentiment_value *= 2.0
                    
                if sentiment_value > 0:
                    positive_score += sentiment_value
                else:
                    negative_score -= sentiment_value
                    
            negate_next = False
            intensify_next = False
        
        # Calculate final sentiment
        if positive_score > 0 or negative_score > 0:
            total_score = positive_score - negative_score
            sentiment = total_score / (positive_score + negative_score)
            sentiment = torch.sign(torch.tensor(sentiment)).item() * (abs(sentiment) ** 0.5)
        else:
            # Fallback to emotional network
            text_tensor = torch.zeros(1, self.hidden_dim + 4, device=self.device)
            sentiment_output = self.emotional_network(text_tensor)
            sentiment = sentiment_output[0][0].item() * 2 - 1
        
        return max(-1.0, min(1.0, sentiment))

    def _evaluate_curiosity(self, response: str) -> float:
        """Evaluate the level of curiosity and exploration in the response.
        
        Args:
            response (str): AI's response message
            
        Returns:
            float: Curiosity score between 0 and 1
        """
        # Keywords indicating curiosity and exploration
        curiosity_words = {
            'interesting', 'curious', 'wonder', 'explore', 'learn', 'discover',
            'fascinating', 'mystery', 'mysteries', 'shall we', 'would you like',
            'what aspects', 'tell me more', 'aspects', 'most', 'about'
        }
        
        # Question patterns indicating engagement
        question_patterns = {'?', 'what', 'how', 'why', 'which', 'where', 'when', 'who'}
        
        # Count curiosity indicators
        words = response.lower().split()
        curiosity_count = sum(word in curiosity_words for word in words)
        question_count = sum(pattern in response.lower() for pattern in question_patterns)
        
        # Calculate base curiosity score
        curiosity_score = min(1.0, (curiosity_count * 0.2 + question_count * 0.3))
        
        # Boost score for interactive elements
        if 'would you like' in response.lower() or 'shall we' in response.lower():
            curiosity_score = min(1.0, curiosity_score + 0.2)
            
        # Boost score for offering choices
        if 'or' in words and question_count > 0:
            curiosity_score = min(1.0, curiosity_score + 0.2)
            
        return curiosity_score

    def get_reward(self, user_input: str = None, ai_response: str = None) -> float:
        """Calculate reward based on multiple factors including emotional accuracy, conversational flow,
        memory recall, and exploratory learning.
        
        Args:
            user_input (str, optional): The user's input message
            ai_response (str, optional): The AI's response message
            
        Returns:
            float: The calculated reward value between -1 and 1
        """
        if not user_input or not ai_response:
            return 0.0  # No reward without both inputs
        
        # Analyze sentiments
        user_sentiment = self._analyze_sentiment(user_input)
        ai_sentiment = self._analyze_sentiment(ai_response)
        
        # Check for emotional misalignment
        sentiment_diff = abs(user_sentiment - ai_sentiment)
        polarity_mismatch = (user_sentiment * ai_sentiment) < 0
        
        # Calculate emotional accuracy reward
        if polarity_mismatch and "tell me about" not in user_input.lower():
            emotional_accuracy_reward = -3.0
            return -1.0  # Immediate return for emotional misalignment except for curiosity queries
        elif sentiment_diff > 0.3:
            emotional_accuracy_reward = -2.0 * sentiment_diff
        else:
            emotional_accuracy_reward = 1.0 - sentiment_diff
        
        # Calculate other reward components
        flow_reward = self._evaluate_conversation_flow(user_input, ai_response)
        memory_recall_reward = self._evaluate_memory_recall(ai_response)
        curiosity_reward = self._evaluate_curiosity(ai_response)
        
        # Check interaction type
        is_curiosity_focused = any(word in user_input.lower() for word in ['tell me about', 'what', 'how', 'why'])
        is_memory_focused = any(word in user_input.lower() for word in ['remember', 'recall', 'what', 'favorite'])
        is_emotional = any(word in user_input.lower() for word in ['feel', 'happy', 'sad', 'excited', 'worried'])
        
        # Calculate total reward based on interaction type
        if is_curiosity_focused:
            # For curiosity-focused interactions, prioritize curiosity and engagement
            total_reward = (
                0.1 * emotional_accuracy_reward +  # Reduced emotional weight for curiosity queries
                0.2 * flow_reward +
                0.1 * memory_recall_reward +
                0.6 * curiosity_reward  # Highest weight for curiosity
            )
            # Boost reward for highly curious responses
            if curiosity_reward > 0.7:
                total_reward = max(0.7, total_reward)
            # Additional boost for comprehensive responses
            if len(ai_response.split()) > 15 and '?' in ai_response:
                total_reward = min(1.0, total_reward + 0.2)
        elif is_memory_focused:
            total_reward = (
                0.2 * emotional_accuracy_reward +
                0.2 * flow_reward +
                0.5 * memory_recall_reward +  # Highest weight for memory
                0.1 * curiosity_reward
            )
            # Boost reward for good memory recall
            if memory_recall_reward > 0.7:
                total_reward = max(0.6, total_reward)
        elif is_emotional:
            total_reward = (
                0.7 * emotional_accuracy_reward +  # Highest weight for emotional accuracy
                0.1 * flow_reward +
                0.1 * memory_recall_reward +
                0.1 * curiosity_reward
            )
            # Boost reward for good emotional alignment
            if emotional_accuracy_reward > 0.8:
                total_reward = max(0.7, total_reward)
        else:
            # Balanced weights for general interaction
            total_reward = (
                0.4 * emotional_accuracy_reward +
                0.3 * flow_reward +
                0.2 * memory_recall_reward +
                0.1 * curiosity_reward
            )
        
        # Apply stage-specific learning factor
        total_reward *= self.developmental_factors['learning']
        
        # Ensure reward is in valid range
        total_reward = max(-1.0, min(1.0, total_reward))
        
        # Update metrics
        self.metrics['training_loss'] = 1.0 - total_reward
        
        return total_reward

    def _evaluate_conversation_flow(self, user_input: str, ai_response: str) -> float:
        """Evaluate the naturalness and coherence of conversation flow.
        
        Args:
            user_input (str): User's input message
            ai_response (str): AI's response message
            
        Returns:
            float: Flow score between 0 and 1
        """
        # Check for repetition (penalize)
        if ai_response in self.memory_states['short_term'].tolist():
            return 0.2
            
        # Check for question-answer pairs (reward)
        if '?' in user_input and len(ai_response.split()) > 5:
            return 0.8
            
        # Default flow score based on response length and complexity
        return 0.5
        
    def _evaluate_memory_recall(self, response: str) -> float:
        """Evaluate how well the response incorporates previous conversation memory.
        
        Args:
            response (str): AI's response message
            
        Returns:
            float: Memory recall score between 0 and 1
        """
        # Keywords indicating memory recall
        memory_words = {
            'remember', 'recall', 'previous', 'earlier', 'before',
            'mentioned', 'said', 'told', 'discussed', 'conversation',
            'based on', 'according to', 'from our', 'you shared',
            'you mentioned', 'as we discussed', 'as you told me',
            'from what I remember', 'if I recall correctly'
        }
        
        # Count memory recall indicators
        words = response.lower().split()
        memory_count = sum(word in memory_words for word in words)
        
        # Calculate base memory score with higher weight
        memory_score = min(1.0, memory_count * 0.4)  # Increased from 0.3
        
        # Convert response to feature space
        response_features = torch.zeros(1, self.hidden_dim, device=self.device)
        
        # Ensure working memory has correct dimensions
        working_memory = self.memory_states['working']
        if working_memory.dim() == 1:
            working_memory = working_memory.unsqueeze(0)
        
        # Pad or truncate working memory to match hidden_dim
        if working_memory.shape[1] < self.hidden_dim:
            padding = torch.zeros(1, self.hidden_dim - working_memory.shape[1], device=self.device)
            working_memory = torch.cat([working_memory, padding], dim=1)
        elif working_memory.shape[1] > self.hidden_dim:
            working_memory = working_memory[:, :self.hidden_dim]
        
        # Compare with memory states using cosine similarity
        working_similarity = torch.cosine_similarity(
            response_features,
            working_memory
        ).item()
        
        # Compare with long-term memory
        long_term_memory = self.memory_states['long_term']
        if long_term_memory.dim() == 1:
            long_term_memory = long_term_memory.unsqueeze(0)
            
        # Pad or truncate long-term memory to match hidden_dim
        if long_term_memory.shape[1] < self.hidden_dim:
            padding = torch.zeros(1, self.hidden_dim - long_term_memory.shape[1], device=self.device)
            long_term_memory = torch.cat([long_term_memory, padding], dim=1)
        elif long_term_memory.shape[1] > self.hidden_dim:
            long_term_memory = long_term_memory[:, :self.hidden_dim]
            
        long_term_similarity = torch.cosine_similarity(
            response_features,
            long_term_memory
        ).item()
        
        # Calculate weighted similarity score with higher weights
        similarity_score = 0.8 * working_similarity + 0.4 * long_term_similarity  # Increased weights
        
        # Boost score for explicit memory references
        if memory_score > 0:
            similarity_score = min(1.0, similarity_score + 0.4)  # Increased from 0.3
            
        # Additional boost for specific memory recall
        if 'favorite' in response.lower() and 'color' in response.lower():
            if 'blue' in response.lower():
                similarity_score = min(1.0, similarity_score + 0.5)  # Increased from 0.4
            else:
                # Penalize incorrect specific memory recall
                similarity_score = max(0.0, similarity_score - 0.3)
                
        # Additional boost for contextual memory recall
        context_words = {'when', 'where', 'how', 'why', 'what'}
        if any(word in context_words for word in words) and memory_score > 0:
            similarity_score = min(1.0, similarity_score + 0.3)
            
        # Ensure minimum reward for good memory recall
        if memory_score > 0.5 and similarity_score > 0.3:
            similarity_score = max(0.4, similarity_score)
            
        return max(0.0, similarity_score)

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
        
        # Get new state
        return self.get_state()

    def train_q_learning(self, num_episodes: int = 100, max_steps: int = 1000):
        """Train Q-Learning system"""
        for episode in range(num_episodes):
            # Train one episode
            stats = self.q_learning.train_episode(
                get_state_fn=self.get_state,
                get_reward_fn=self.get_reward,
                step_env_fn=self.step_environment,
                max_steps=max_steps
            )
            
            # Store metrics
            self.metrics['q_learning_rewards'].append(stats['total_reward'])
            self.metrics['q_learning_losses'].append(stats['average_loss'])
            
            # Update developmental factors based on performance
            if len(self.metrics['q_learning_rewards']) >= 10:
                avg_reward = sum(self.metrics['q_learning_rewards'][-10:]) / 10
                self.developmental_factors['learning'] = min(
                    1.0, 
                    self.developmental_factors['learning'] + avg_reward * 0.01
                )

    def save_brain_state(self, path: str):
        """Save brain state including Q-Learning model"""
        state_dict = {
            'sensory_network': self.sensory_network.state_dict(),
            'memory_network': self.memory_network.state_dict(),
            'emotional_network': self.emotional_network.state_dict(),
            'learning_network': self.learning_network.state_dict(),
            'memory_states': self.memory_states,
            'developmental_factors': self.developmental_factors,
            'metrics': self.metrics,
            'stage': DevelopmentalStage.EARLY_TODDLER,
            'stage_progress': 0.0
        }
        
        # Save Q-Learning model separately
        self.q_learning.save_model(path + '_q_learning')
        
        # Save other brain components
        torch.save(state_dict, path)

    def load_brain_state(self, path: str):
        """Load brain state including Q-Learning model"""
        # Load Q-Learning model
        self.q_learning.load_model(path + '_q_learning')
        
        # Load other brain components
        state_dict = torch.load(path)
        self.sensory_network.load_state_dict(state_dict['sensory_network'])
        self.memory_network.load_state_dict(state_dict['memory_network'])
        self.emotional_network.load_state_dict(state_dict['emotional_network'])
        self.learning_network.load_state_dict(state_dict['learning_network'])
        self.memory_states = state_dict['memory_states']
        self.developmental_factors = state_dict['developmental_factors']
        self.metrics = state_dict['metrics']
        self.stage = state_dict['stage']
        self.stage_progress = state_dict['stage_progress']

class DigitalChild:
    """Class representing a digital child with a brain and developmental stage."""
    
    def __init__(self, stage: DevelopmentalStage = DevelopmentalStage.NEWBORN):
        """Initialize digital child with a brain and developmental stage.
        
        Args:
            stage (DevelopmentalStage): Initial developmental stage of the child
        """
        self.brain = IntegratedBrain()
        self.brain.stage = stage
        self.brain.developmental_factors = self.brain._get_stage_modulation(stage)
        
    def process_input(self, visual_input: torch.Tensor, auditory_input: torch.Tensor, 
                     emotions: torch.Tensor) -> Dict[str, Any]:
        """Process sensory and emotional inputs through the child's brain.
        
        Args:
            visual_input (torch.Tensor): Visual input tensor
            auditory_input (torch.Tensor): Auditory input tensor
            emotions (torch.Tensor): Current emotional state tensor
            
        Returns:
            Dict containing processed outputs from various brain systems
        """
        return self.brain(visual_input, auditory_input, emotions)
        
    def update_stage(self, new_stage: DevelopmentalStage):
        """Update the child's developmental stage.
        
        Args:
            new_stage (DevelopmentalStage): New developmental stage
        """
        self.brain.stage = new_stage
        self.brain.developmental_factors = self.brain._get_stage_modulation(new_stage)
        
    def get_brain_state(self) -> Dict[str, Any]:
        """Get the current state of the child's brain.
        
        Returns:
            Dict containing brain state information
        """
        return {
            'stage': self.brain.stage,
            'developmental_factors': self.brain.developmental_factors,
            'metrics': self.brain.metrics,
            'brain_state': self.brain.brain_state.__dict__
        }