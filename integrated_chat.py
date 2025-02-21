# integrated_chat.py
# Description: Master chat interface integrating all neural networks and systems
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from conversation_system import ConversationSystem
from emotional_memory_system import EmotionalMemorySystem, EmotionalMemoryEntry
from neural_architecture import NeuralArchitecture
from decision_network import DecisionNetwork
from ollama_chat import OllamaChat
from obsidian_api import ObsidianAPI
from emotional_regulation import EmotionalRegulation
from memory_module import MemoryModule
from self_supervised_trainer import SelfSupervisedTrainer
from moral_network import MoralNetwork
from rag_memory import RAGMemorySystem
from self_awareness_network import SelfAwarenessNetwork
from integrated_brain import IntegratedBrain
from developmental_stages import DevelopmentalStage
from logger import DevelopmentLogger
from brain_state import BrainState

class IntegratedChatSystem:
    """Master chat interface integrating all neural networks and systems"""
    
    def __init__(self, initial_stage: DevelopmentalStage = DevelopmentalStage.NEWBORN):
        """Initialize the integrated chat system"""
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize brain state
        self.brain_state = BrainState()
        self.brain_state.emotional_valence = 0.5
        self.brain_state.arousal = 0.3
        self.brain_state.attention = 0.7
        self.brain_state.consciousness = 0.8
        self.brain_state.stress = 0.2
        self.brain_state.fatigue = 0.1
        self.brain_state.neurotransmitters = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'norepinephrine': 0.5,
            'gaba': 0.5,
            'glutamate': 0.5
        }
        
        # Initialize emotional memory parameters
        self.emotional_decay_rate = 0.85  # Increased decay rate for more dynamic changes
        self.emotional_impact_factor = 0.4  # Increased impact factor
        self.emotional_recovery_rate = 0.1  # New: Rate at which emotions recover
        self.last_emotion_update = datetime.now()
        
        # Initialize development tracking
        self.emotional_maturity = 0.1
        self.learning_rate = 0.05  # Increased learning rate
        self.max_emotional_change = 0.3  # New: Maximum change per interaction
        
        # Initialize brain
        self.brain = IntegratedBrain(stage=initial_stage).to(self.device)
        self.brain.brain_state = self.brain_state
        
        # Initialize neural networks
        self.decision_network = DecisionNetwork().to(self.device)
        self.self_awareness = SelfAwarenessNetwork().to(self.device)
        self.moral_network = MoralNetwork().to(self.device)
        
        # Initialize ChromaDB
        persist_dir = Path("chat_memory")
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize collections
        collections_to_init = [
            ("conversation", "Stores conversation history"),
            ("vocabulary", "Stores known words and their embeddings"),
            ("dream", "Stores generated dreams"),
            ("emotional_memory", "Stores emotional memories"),
            ("episodic_memory", "Stores episodic memories"),
            ("semantic_memory", "Stores semantic memories")
        ]

        # Delete existing collections
        for collection_name, _ in collections_to_init:
            try:
                self.chroma_client.delete_collection(collection_name)
            except ValueError:
                pass

        # Create all collections
        for collection_name, description in collections_to_init:
            setattr(
                self,
                f"{collection_name}_collection",
                self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": description}
                )
            )

        # Initialize memory systems
        self.emotional_memory = EmotionalMemorySystem(
            chat_interface=self,
            sentence_transformer_model="all-MiniLM-L6-v2"
        )
        self.rag_memory = RAGMemorySystem(
            chat_interface=self,
            sentence_transformer_model="all-MiniLM-L6-v2"
        )
        self.memory_module = MemoryModule()
        
        # Initialize logger
        self.logger = DevelopmentLogger()
        
        # Initialize language and conversation systems
        self.conversation_system = ConversationSystem(logger=self.logger)
        self.ollama = OllamaChat(model="artifish/llama3.2-uncensored")
        
        # Initialize emotional regulation
        self.emotional_regulation = EmotionalRegulation(logger=self.logger)
        
        # Initialize self-supervised learning
        self.self_supervised = SelfSupervisedTrainer(model=self.brain)
        
        # Initialize Obsidian integration
        obsidian_vault_path = Path("neural_child_vault")
        obsidian_vault_path.mkdir(parents=True, exist_ok=True)
        self.obsidian = ObsidianAPI(vault_path=str(obsidian_vault_path))
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Track conversation state
        self.conversation_history = []
        self.last_interaction = datetime.now()
        self.message_counter = 0
        
        # Initialize development tracking
        self.stage = initial_stage
        self.stage_progress = 0.0
        self.creation_time = datetime.now()
        
        # Create necessary directories
        Path("emotional_memories").mkdir(parents=True, exist_ok=True)
        Path("conversation_history").mkdir(parents=True, exist_ok=True)
        Path("brain_states").mkdir(parents=True, exist_ok=True)
        
        # Initialize dimension projection
        self.input_projection = nn.Linear(384, 512).to(self.device)
        
    def _update_emotional_state(self, message_sentiment: float, message_intensity: float):
        """Update emotional state based on interaction"""
        # Calculate time since last update for decay
        time_delta = (datetime.now() - self.last_emotion_update).total_seconds()
        decay_factor = self.emotional_decay_rate ** (time_delta / 60)  # Decay based on minutes
        
        # Get recent emotional history
        recent_states = []
        if len(self.conversation_history) > 0:
            recent_states = [entry['brain_state'] for entry in self.conversation_history[-3:]]
        
        # Calculate emotional volatility
        emotional_volatility = 0.0
        if len(recent_states) >= 2:
            changes = [abs(s['emotional_valence'] - p['emotional_valence']) 
                      for s, p in zip(recent_states[1:], recent_states[:-1])]
            emotional_volatility = sum(changes) / len(changes)
        
        # Calculate base emotional impact with context awareness
        base_impact = message_sentiment * self.emotional_impact_factor
        if message_sentiment > 0:
            if emotional_volatility > 0.3:  # High volatility reduces positive impact
                base_impact *= 0.5  # Reduced positive response due to uncertainty
            else:
                base_impact *= 1.5  # Normal amplification of positive emotions
        
        maturity_factor = 1.0 - (self.emotional_maturity * 0.3)
        
        # Update emotional valence with smoother transitions
        if message_sentiment > 0:
            # Positive interaction increases joy and trust, reduces fear
            if emotional_volatility < 0.3:  # Normal positive response
                self.brain_state.emotional_valence = min(0.9, 
                    self.brain_state.emotional_valence * 0.7 + (0.3 + base_impact * maturity_factor))
                self.brain_state.neurotransmitters['dopamine'] = min(0.9,
                    self.brain_state.neurotransmitters['dopamine'] * 0.8 + 0.3)
                self.brain_state.neurotransmitters['serotonin'] = min(0.9,
                    self.brain_state.neurotransmitters['serotonin'] * 0.8 + 0.25)
                self.brain_state.stress = max(0.1,
                    self.brain_state.stress * 0.6)  # Faster stress reduction
            else:  # Cautious positive response due to volatility
                self.brain_state.emotional_valence = min(0.9,
                    self.brain_state.emotional_valence * 0.9 + (0.1 + base_impact * maturity_factor))
                self.brain_state.neurotransmitters['dopamine'] = min(0.9,
                    self.brain_state.neurotransmitters['dopamine'] * 0.9 + 0.1)
                self.brain_state.neurotransmitters['serotonin'] = min(0.9,
                    self.brain_state.neurotransmitters['serotonin'] * 0.9 + 0.1)
                self.brain_state.stress = min(0.9,
                    self.brain_state.stress + 0.1)  # Slight stress increase from uncertainty
        else:
            # Negative interaction decreases joy and trust, increases stress
            self.brain_state.emotional_valence = max(0.1,
                self.brain_state.emotional_valence * 0.8 + (base_impact * maturity_factor))
            self.brain_state.neurotransmitters['dopamine'] = max(0.1,
                self.brain_state.neurotransmitters['dopamine'] * 0.9)
            self.brain_state.neurotransmitters['serotonin'] = max(0.1,
                self.brain_state.neurotransmitters['serotonin'] * 0.95)
            self.brain_state.stress = min(0.9,
                self.brain_state.stress * 0.8 + (abs(base_impact) * 0.3))
            
        # Update arousal based on intensity and emotional state
        target_arousal = min(0.9, max(0.1,
            self.brain_state.arousal * decay_factor + 
            message_intensity * 0.3 + 
            emotional_volatility * 0.2 +  # Higher volatility increases arousal
            abs(base_impact) * 0.2))
        self.brain_state.arousal = target_arousal
        
        # Update consciousness more dynamically
        target_consciousness = (
            self.brain_state.arousal +
            self.brain_state.attention +
            self.brain_state.neurotransmitters['dopamine'] +
            (1.0 - self.brain_state.stress) +  # Higher stress reduces consciousness
            self.brain_state.neurotransmitters['serotonin'] * 0.5 +  # Trust influences consciousness
            (1.0 - emotional_volatility) * 0.3  # Stability improves consciousness
        ) / 5.0  # Adjusted divisor for new factors
        
        self.brain_state.consciousness = min(0.9, max(0.1,
            self.brain_state.consciousness * 0.7 + target_consciousness * 0.3))
        
        # Update emotional maturity with context
        if message_sentiment > 0 and emotional_volatility < 0.3:
            self.emotional_maturity = min(1.0,
                self.emotional_maturity + self.learning_rate * 0.1)
        elif emotional_volatility > 0.3:
            self.emotional_maturity = min(1.0,
                self.emotional_maturity + self.learning_rate * 0.05)  # Slower growth during volatility
        
        # Update timestamp
        self.last_emotion_update = datetime.now()
        
    def _update_memory_with_emotion(self, message: str, emotion: float, intensity: float):
        """Store memory with emotional context"""
        try:
            # Create flattened metadata for ChromaDB
            brain_state = self.get_brain_state()
            flat_metadata = {
                'age_months': self._get_age_months(),
                'stage': self.stage.name,
                'emotional_maturity': float(self.emotional_maturity),
                'emotional_valence': float(brain_state['emotional_valence']),
                'arousal': float(brain_state['arousal']),
                'attention': float(brain_state['attention']),
                'consciousness': float(brain_state['consciousness']),
                'stress': float(brain_state['stress']),
                'dopamine': float(brain_state['neurotransmitters']['dopamine']),
                'serotonin': float(brain_state['neurotransmitters']['serotonin']),
                'norepinephrine': float(brain_state['neurotransmitters']['norepinephrine']),
                'intensity': float(abs(intensity)),
                'valence': float(emotion),
                'timestamp': datetime.now().isoformat(),
                'content': message  # Add content to metadata
            }
            
            # Store in ChromaDB emotional memory collection
            memory_id = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.emotional_memory_collection.add(
                documents=[message],
                metadatas=[flat_metadata],
                ids=[memory_id]
            )
            
            # Store full data structure in Obsidian
            self.obsidian.create_note(
                title=f"Memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                content=f"""# Emotional Memory Entry

## Message
{message}

## Emotional Context
- Valence: {emotion:.2f}
- Intensity: {intensity:.2f}
- Arousal: {self.brain_state.arousal:.2f}

## Brain State
{json.dumps(self.get_brain_state(), indent=2)}

## Metadata
- Age: {self._get_age_months()} months
- Stage: {self.stage.name}
- Emotional Maturity: {self.emotional_maturity:.2f}
""",
                folder="Memories"
            )
            
            # Store in RAG memory with all required parameters
            self.rag_memory.store_memory(
                content=message,
                memory_type='emotional',
                emotional_state=flat_metadata,
                brain_state=brain_state,  # Add brain_state parameter
                metadata=flat_metadata
            )
            
        except Exception as e:
            print(f"Error storing memory: {str(e)}")

    def get_emotional_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Get emotional memories related to query"""
        try:
            # Search ChromaDB for relevant memories
            results = self.emotional_memory_collection.query(
                query_texts=[query],
                n_results=min(limit, max(1, len(self.conversation_history)))  # Adjust limit based on available memories
            )
            
            memories = []
            if results and results['metadatas']:
                for metadata in results['metadatas'][0]:
                    if isinstance(metadata, dict) and 'content' in metadata:
                        memories.append(metadata)
            
            return memories
            
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []
        
    def process_message(self, message: str) -> str:
        """Process incoming message through all systems"""
        try:
            # Get message embedding and ensure float32 dtype
            message_embedding = self.sentence_transformer.encode(message, convert_to_tensor=True)
            message_embedding = message_embedding.to(self.device).float()
            
            # Process through brain
            projected_input = self.input_projection(message_embedding.unsqueeze(0))
            brain_output = self.brain(projected_input)
            
            # Extract emotional signals with stronger sentiment
            emotional_output = brain_output['emotional'].float()
            sentiment = float(emotional_output[0][0].item()) * 2.0 - 1.0  # Convert to [-1, 1] range
            intensity = float(emotional_output[0][1].item()) * 1.5  # Increased intensity
            
            # Update emotional state
            self._update_emotional_state(sentiment, intensity)
            
            # Update memory with emotional context
            self._update_memory_with_emotion(message, sentiment, intensity)
            
            # Record interaction in Obsidian
            self.obsidian.create_note(
                title=f"Interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                content=f"""
# Interaction Record

## Message
{message}

## Emotional State
- Valence: {self.brain_state.emotional_valence:.2f}
- Arousal: {self.brain_state.arousal:.2f}
- Attention: {self.brain_state.attention:.2f}
- Consciousness: {self.brain_state.consciousness:.2f}
- Stress: {self.brain_state.stress:.2f}

## Neurotransmitters
- Dopamine: {self.brain_state.neurotransmitters['dopamine']:.2f}
- Serotonin: {self.brain_state.neurotransmitters['serotonin']:.2f}
- Norepinephrine: {self.brain_state.neurotransmitters['norepinephrine']:.2f}

## Development Stage
- Current Stage: {self.stage.name}
- Age (months): {self._get_age_months()}
- Progress: {self.stage_progress:.2f}

## Vocabulary
{self._extract_vocabulary(message)}
"""
            )
            
            # Process through decision network with updated emotional state
            emotional_state = torch.tensor([[
                self.brain_state.emotional_valence,
                self.brain_state.arousal,
                self.brain_state.attention,
                self.brain_state.consciousness
            ]], device=self.device, dtype=torch.float32)
            
            # Create a sequence of embeddings for conversation context
            conversation_embeddings = message_embedding.unsqueeze(0).unsqueeze(0)
            memory_context = message_embedding.unsqueeze(0)
            
            decision_output = self.decision_network(
                conversation_embeddings,
                emotional_state,
                memory_context
            )
            
            # Process through self-awareness network
            self_awareness_output = self.self_awareness(message_embedding.unsqueeze(0))
            
            # Process through moral network
            moral_output = self.moral_network(message_embedding.unsqueeze(0))
            
            # Update conversation history with new state
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat(),
                'brain_state': self.get_brain_state()
            })
            
            # Retrieve relevant memories
            relevant_memories = self.get_emotional_memories(message)
            memory_context = ""
            if relevant_memories:
                memory_context = "\nPrevious relevant memories:\n"
                for memory in relevant_memories:
                    if isinstance(memory, dict) and 'content' in memory:
                        memory_context += f"- {memory['content']}\n"
                        memory_context += f"  (Valence: {memory.get('valence', 0.0):.2f}, "
                        memory_context += f"Trust: {memory.get('serotonin', 0.0):.2f}, "
                        memory_context += f"Fear: {memory.get('stress', 0.0):.2f})\n"
            
            # Generate response using Ollama with updated state and memory context
            response = self.ollama.generate_response(
                message=message + memory_context,  # Include memory context
                emotional_state=self.get_brain_state(),
                brain_state={
                    'sensory': brain_output['sensory'].detach().cpu().numpy().tolist(),
                    'memory': brain_output['memory'].detach().cpu().numpy().tolist(),
                    'emotional': brain_output['emotional'].detach().cpu().numpy().tolist(),
                    'decision': brain_output['decision'].detach().cpu().numpy().tolist()
                },
                stage=self.stage.name,
                age_months=self._get_age_months()
            )
            
            # Process response through emotional regulation with updated state
            regulated_response = self.emotional_regulation.regulate_emotion(
                torch.tensor([
                    self.brain_state.emotional_valence,
                    self.brain_state.arousal,
                    self.brain_state.attention,
                    self.brain_state.consciousness
                ], device=self.device, dtype=torch.float32)
            )
            
            # Update conversation history with response and final state
            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat(),
                'brain_state': self.get_brain_state()
            })
            
            # Format final response with updated emotional state
            final_response = self._format_response(response)
            
            # Update last interaction time
            self.last_interaction = datetime.now()
            
            return final_response
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return "I'm not working right now. Fix me."
            
    def _format_response(self, response: str) -> str:
        """Format response with emotional state and brain state"""
        brain_state = self.brain.get_brain_state()
        
        formatted = f"Child: {response}\n\n"
        formatted += "Emotional State:\n"
        formatted += f"Joy: {brain_state['emotional_valence']:.2f}\n"
        formatted += f"Trust: {brain_state['neurotransmitters']['serotonin']:.2f}\n"
        formatted += f"Fear: {brain_state['stress']:.2f}\n"
        formatted += f"Arousal: {brain_state['arousal']:.2f}\n"
        formatted += f"Attention: {brain_state['attention']:.2f}\n"
        formatted += f"Consciousness: {brain_state['consciousness']:.2f}\n"
        
        return formatted
        
    def _get_age_months(self) -> int:
        """Calculate age in months since creation"""
        now = datetime.now()
        delta = now - self.creation_time
        return int(delta.days / 30.44)  # Average month length
        
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state"""
        return {
            'emotional_valence': float(self.brain_state.emotional_valence),
            'arousal': float(self.brain_state.arousal),
            'attention': float(self.brain_state.attention),
            'consciousness': float(self.brain_state.consciousness),
            'stress': float(self.brain_state.stress),
            'fatigue': float(self.brain_state.fatigue),
            'neurotransmitters': {
                k: float(v) for k, v in self.brain_state.neurotransmitters.items()
            }
        }
        
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
        
    def save_state(self, path: str = "brain_states"):
        """Save current brain state"""
        state = {
            'brain_state': self.brain.get_brain_state(),
            'stage': self.stage.name,
            'stage_progress': self.stage_progress,
            'creation_time': self.creation_time.isoformat(),
            'last_interaction': self.last_interaction.isoformat()
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = Path(path) / f"brain_state_{timestamp}.json"
        
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, state_file: str):
        """Load brain state from file"""
        with open(state_file, 'r') as f:
            state = json.load(f)
            
        self.stage = DevelopmentalStage[state['stage']]
        self.stage_progress = state['stage_progress']
        self.creation_time = datetime.fromisoformat(state['creation_time'])
        self.last_interaction = datetime.fromisoformat(state['last_interaction'])
        
        # Update brain state
        self.brain.brain_state.emotional_valence = state['brain_state']['emotional_valence']
        self.brain.brain_state.arousal = state['brain_state']['arousal']
        self.brain.brain_state.attention = state['brain_state']['attention']
        self.brain.brain_state.consciousness = state['brain_state']['consciousness']
        self.brain.brain_state.stress = state['brain_state']['stress']
        self.brain.brain_state.fatigue = state['brain_state']['fatigue']
        
        for nt, value in state['brain_state']['neurotransmitters'].items():
            self.brain.brain_state.neurotransmitters[nt] = value 

    def _extract_vocabulary(self, message: str) -> str:
        """Extract and analyze vocabulary from message"""
        words = message.lower().split()
        unique_words = set(words)
        
        # Store in vocabulary collection
        for word in unique_words:
            self.vocabulary_collection.add(
                documents=[word],
                metadatas=[{
                    'first_seen': datetime.now().isoformat(),
                    'context': message,
                    'age_months': self._get_age_months(),
                    'stage': self.stage.name
                }],
                ids=[f"word_{word}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
            )
        
        return f"""
### New Words Learned
{', '.join(unique_words)}

### Context
{message}
""" 