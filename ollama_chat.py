# ollama_chat.py
# Description: Advanced chat system for neural child development using Ollama
# Created by: Christopher Celaya

import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer
from llm_module import chat_completion
from memory_store import MemoryStore
from language_development import LanguageDevelopment
from neural_architecture import BrainState
import re

class OllamaChildChat:
    def __init__(
        self,
        memory_store: MemoryStore,
        emotional_system,  # This will be the DigitalChild instance
        language_system: LanguageDevelopment,
        model_name: str = "artifish/llama3.2-uncensored"
    ):
        self.memory_store = memory_store
        self.emotional_system = emotional_system
        self.language_system = language_system
        self.model_name = model_name
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.conversation_history: List[Dict] = []
        self.context_window = 5  # Reduced for newborn stage
        self.name = "Alex"  # Give the child a consistent name
        self.known_names = {}  # Store known names and relationships
        
    def _generate_system_prompt(self) -> str:
        """Generate the system prompt based on current development state."""
        # Get brain state
        brain_state = self.emotional_system.brain.get_brain_state()
        
        # Get emotional state values
        joy = float(self.emotional_system.emotional_state[0])
        trust = float(self.emotional_system.emotional_state[1])
        fear = float(self.emotional_system.emotional_state[2])
        surprise = float(self.emotional_system.emotional_state[3])
        
        current_feeling = self.emotional_system.express_feeling()
        
        # Get known names for context
        names_context = ""
        if self.known_names:
            names_context = "\nI know these people:\n"
            for name, relation in self.known_names.items():
                names_context += f"- {name} is {relation}\n"
        
        # Create detailed prompt based on brain and emotional state
        prompt = f"""You are a neural child named {self.name} in development. Your current state is:

Stage: {self.emotional_system.curriculum.current_stage.name}
Age: {self.emotional_system.age()} months
Language Stage: {self.language_system.get_current_stage()}

Brain State:
- Arousal: {brain_state['arousal']:.2f}
- Attention: {brain_state['attention']:.2f}
- Emotional Valence: {brain_state['emotional_valence']:.2f}
- Consciousness: {brain_state['consciousness']:.2f}

Emotional State:
- Joy: {joy:.2f}
- Trust: {trust:.2f}
- Fear: {fear:.2f}
- Surprise: {surprise:.2f}
Current Feeling: {current_feeling}{names_context}

Previous messages for context:
{self._format_conversation_history()}

Respond naturally as {self.name}, showing:
1. Age-appropriate emotional responses
2. Current development stage behaviors
3. Learning and curiosity
4. Emotional regulation based on brain state
5. Language abilities matching your stage
6. Consistent memory of people you know

Format your response as JSON with:
{{
    "response": "your natural response",
    "emotions": {{
        "joy": 0.0-1.0,
        "trust": 0.0-1.0,
        "fear": 0.0-1.0,
        "surprise": 0.0-1.0
    }},
    "learning": {{
        "new_words": ["list", "of", "words"],
        "concepts": ["list", "of", "concepts"]
    }},
    "memory": {{
        "type": "emotional/semantic/episodic",
        "content": "memory content",
        "emotional_value": 0.0-1.0
    }}
}}"""
        
        return prompt

    def _format_conversation_history(self) -> str:
        """Format recent conversation history for context"""
        if not self.conversation_history:
            return "No previous messages"
            
        history = []
        for msg in self.conversation_history[-self.context_window:]:
            history.append(f"User: {msg['user']}")
            if 'response' in msg:
                history.append(f"{self.name}: {msg['response']}")
                
        return "\n".join(history)

    def _extract_learnable_content(self, message: str) -> Tuple[List[str], List[str]]:
        """Extract potential new words and concepts from the message."""
        # Use the language system to identify new words
        new_words = self.language_system.identify_new_words(message)
        
        # Use the language system to identify new concepts
        new_concepts = self.language_system.identify_new_concepts(message)
        
        return new_words, new_concepts

    def _create_memory(self, content: str, memory_type: str, emotional_value: float):
        """Create and store a new memory."""
        embedding = self.sentence_encoder.encode(content).tolist()
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "emotional_value": emotional_value,
            "type": memory_type,
            "memory_type": memory_type,  # Explicitly set memory type
            "age_months": self.emotional_system.age(),
            "stage": self.emotional_system.curriculum.current_stage.name,
            "feeling": self.emotional_system.express_feeling()
        }
        
        # Get current emotional state
        emotional_state = {
            "joy": float(self.emotional_system.emotional_state[0]),
            "trust": float(self.emotional_system.emotional_state[1]),
            "fear": float(self.emotional_system.emotional_state[2]),
            "surprise": float(self.emotional_system.emotional_state[3])
        }
        
        try:
            # Store memory based on type
            if memory_type == "episodic":
                self.memory_store.store_episodic_memory(
                    content=content,
                    embedding=embedding,
                    metadata=metadata
                )
            elif memory_type == "semantic":
                self.memory_store.store_semantic_memory(
                    content=content,
                    embedding=embedding,
                    metadata=metadata
                )
            else:  # emotional or unknown type
                self.memory_store.store_emotional_memory(
                    content=content,
                    embedding=embedding,
                    emotional_state=emotional_state,
                    metadata=metadata
                )
        except Exception as e:
            print(f"Error storing memory: {str(e)}")
            # If storing fails, try as emotional memory
            try:
                self.memory_store.store_emotional_memory(
                    content=content,
                    embedding=embedding,
                    emotional_state=emotional_state,
                    metadata=metadata
                )
            except Exception as e2:
                print(f"Error storing fallback emotional memory: {str(e2)}")

    def _update_emotional_state(self, emotions: Dict[str, float]):
        """Update the emotional state based on the interaction."""
        # Convert emotions dict to tensor
        emotion_vector = torch.tensor([
            emotions.get('joy', 0.5),
            emotions.get('trust', 0.5),
            emotions.get('fear', 0.2),
            emotions.get('surprise', 0.3)
        ], device=self.emotional_system.emotional_state.device)
        
        # Update the child's emotional state
        self.emotional_system.update_emotions(emotion_vector)

    def _get_relevant_memories(self, message: str, limit: int = 3) -> List[Dict]:
        """Retrieve relevant memories for context."""
        try:
            # Get emotional memories
            emotional_memories = self.emotional_system.get_emotional_memories(query=message)
            
            # Get semantic memories
            semantic_memories = self.memory_store.retrieve_similar_memories(message, k=limit)
            
            # Combine and sort by relevance/recency
            all_memories = []
            
            # Add emotional memories with type
            for mem in emotional_memories:
                all_memories.append({
                    'content': mem['content'],
                    'type': 'emotional',
                    'metadata': mem['metadata'],
                    'similarity': mem.get('similarity_score', 0)
                })
                
            # Add semantic memories with type
            for mem in semantic_memories:
                all_memories.append({
                    'content': mem['content'],
                    'type': 'semantic',
                    'metadata': mem.get('metadata', {}),
                    'similarity': 0.5  # Default similarity for semantic memories
                })
                
            # Sort by similarity and take top results
            all_memories.sort(key=lambda x: x['similarity'], reverse=True)
            return all_memories[:limit]
            
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []

    def _process_name_learning(self, message: str):
        """Process message for name learning."""
        # Common name introduction patterns
        name_patterns = [
            r"(?i)my name is (\w+)",
            r"(?i)i(?:'|a)m (\w+)",
            r"(?i)call me (\w+)",
            r"(?i)this is (\w+)",
            r"(?i)!teach (?:your |my )?name\s*:\s*(\w+)"  # Handle !teach command
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message)
            if match:
                name = match.group(1).lower()
                if name == "alex":
                    # Learning own name
                    self._create_memory(
                        content=f"I learned that my name is Alex",
                        memory_type="semantic",
                        emotional_value=0.9  # High emotional value for own name
                    )
                    return "self"
                elif name not in self.known_names:
                    self.known_names[name] = "someone I just met"
                    # Create a memory of learning the name
                    self._create_memory(
                        content=f"I learned that someone's name is {name}",
                        memory_type="semantic",
                        emotional_value=0.6
                    )
                return name
        return None

    def chat(self, message: str) -> Dict[str, Any]:
        """Process chat message and generate response."""
        try:
            # Process name learning
            learned_name = self._process_name_learning(message)
            
            # Store message in history
            self.conversation_history.append({
                'user': message,
                'timestamp': datetime.now().isoformat(),
                'learned_name': learned_name
            })
            
            # Get message embedding
            message_embedding = self.sentence_encoder.encode(message, convert_to_tensor=True)
            
            # Create simulated auditory input
            auditory_input = message_embedding.clone()
            
            # Process through integrated brain architecture
            brain_output = self.emotional_system.brain(
                visual_input=message_embedding,
                auditory_input=auditory_input,
                current_emotions=self.emotional_system.emotional_state
            )
            
            # Extract emotional output
            emotional_output = brain_output['emotional']
            new_emotions = emotional_output['emotions']
            
            # Update emotional state with momentum
            self.emotional_system.update_emotions(new_emotions)
            
            # Get relevant memories for context
            memories = self._get_relevant_memories(message)
            memory_context = "\n".join([
                f"{mem['type'].capitalize()} memory: {mem['content']}"
                for mem in memories
            ])
            
            # Build prompt with emotional context
            prompt = self._build_prompt(
                message=message,
                memories=memories,
                emotional_state={
                    'joy': float(new_emotions[0]),
                    'trust': float(new_emotions[1]),
                    'fear': float(new_emotions[2]),
                    'surprise': float(new_emotions[3])
                },
                brain_state=brain_output
            )
            
            # Get response from Ollama
            response = self._get_ollama_response(prompt)
            
            if response and isinstance(response, dict):
                # Update conversation history with response
                self.conversation_history[-1]['response'] = response.get('response', '')
                
                # Update emotional state based on response
                if 'emotions' in response:
                    self._process_response_emotions(response)
                
                # Create memory of interaction
                memory_result = self._create_interaction_memory(
                    message=message,
                    response=response,
                    emotional_state={
                        'joy': float(new_emotions[0]),
                        'trust': float(new_emotions[1]),
                        'fear': float(new_emotions[2]),
                        'surprise': float(new_emotions[3])
                    },
                    brain_output=brain_output
                )
                
                # Format brain state for output
                formatted_brain_state = {
                    'arousal': brain_output['sensory']['sensory_features'].mean().item(),
                    'attention': brain_output['decision']['action'].mean().item(),
                    'emotional_valence': (new_emotions[0] + new_emotions[1] - new_emotions[2]) / 2,
                    'consciousness': brain_output['integrated'].mean().item(),
                    'neurotransmitters': {
                        'dopamine': min(1.0, 0.5 + float(new_emotions[0]) * 0.3),
                        'serotonin': min(1.0, 0.5 + float(new_emotions[1]) * 0.2),
                        'norepinephrine': min(1.0, float(new_emotions[3])),
                        'gaba': min(1.0, 1.0 - float(new_emotions[2]) * 0.5),
                        'glutamate': min(1.0, float(new_emotions[3]))
                    }
                }
                
                # Update development progress
                progress_data = {
                    'emotional_state': response.get('emotions', {}),
                    'vocabulary_size': len(self.language_system.vocabulary),
                    'interaction_quality': memory_result.get('emotional_value', 0.5)
                }
                
                # Check for stage advancement
                level_up = self.emotional_system.curriculum.update_progress(progress_data)
                
                result = {
                    'response': response.get('response', ''),
                    'emotions': response.get('emotions', {}),
                    'learning': {
                        'new_words': self.language_system.extract_new_words(message),
                        'concepts': response.get('learning', {}).get('concepts', [])
                    },
                    'memory': memory_result,
                    'brain_state': formatted_brain_state,
                    'integrated_features': {
                        'visual_cognitive': brain_output['sensory']['sensory_features'].mean().item(),
                        'emotional_response': emotional_output['modulated_features'].mean().item(),
                        'memory_formation': brain_output['memory']['integrated_memory'].mean().item(),
                        'decision_making': brain_output['decision']['action'].mean().item(),
                        'learning_progress': brain_output['integrated'].mean().item()
                    }
                }
                
                # Add level up notification if applicable
                if level_up:
                    result['level_up'] = level_up
                    print("\n" + "="*80)
                    print(level_up['message'])
                    print("\nAchievements:")
                    for achievement in level_up['achievements']:
                        print(f"  {achievement}")
                    print("="*80 + "\n")
                
                return result
            
            return None
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return None

    def _build_prompt(self, message: str, memories: List[Dict], 
                     emotional_state: Dict[str, float],
                     brain_state: Dict[str, Any]) -> str:
        """Build prompt with emotional context and memories."""
        # Format emotional state
        emotion_str = " ".join([
            f"{k.capitalize()}: {v:.2f}"
            for k, v in emotional_state.items()
        ])
        
        # Format brain state
        brain_str = f"""
Brain State:
- Arousal: {brain_state['sensory']['sensory_features'].mean().item():.2f}
- Attention: {brain_state['decision']['action'].mean().item():.2f}
- Emotional Processing: {brain_state['emotional']['modulated_features'].mean().item():.2f}
- Memory Integration: {brain_state['memory']['integrated_memory'].mean().item():.2f}
- Learning Progress: {brain_state['integrated'].mean().item():.2f}

Neural Activity:
- Sensory Processing: {brain_state['sensory']['sensory_features'].mean().item():.2f}
- Memory Formation: {brain_state['memory']['working_memory'].mean().item():.2f}
- Decision Making: {brain_state['decision']['action'].mean().item():.2f}
- Emotional Response: {brain_state['emotional']['emotions'].mean().item():.2f}
"""
        
        # Format memories with emotional context
        memory_str = ""
        for mem in memories:
            memory_emotion = mem.get('metadata', {}).get('emotional_state', {})
            memory_feeling = mem.get('feeling_at_time', 'Unknown feeling')
            memory_str += f"\n- {mem['type'].capitalize()}: {mem['content']}"
            memory_str += f"\n  When this happened I felt {memory_feeling} with:"
            memory_str += f"\n  Joy: {memory_emotion.get('joy', 0):.2f}"
            memory_str += f"\n  Trust: {memory_emotion.get('trust', 0):.2f}"
            memory_str += f"\n  Fear: {memory_emotion.get('fear', 0):.2f}"
            memory_str += f"\n  Surprise: {memory_emotion.get('surprise', 0):.2f}"
        
        if not memories:
            memory_str = "\nNo relevant memories found"
        
        # Build complete prompt
        prompt = f"""You are {self.name}, a neural child with:

Emotional State:
{emotion_str}

{brain_str}

Relevant memories and how they made you feel:{memory_str}

User message: {message}

Remember:
1. You are {self.name}, not Chris or any other name
2. Maintain emotional continuity with your memories
3. React naturally to the user's message based on your emotional state and memories
4. Show appropriate emotional regulation for your development stage
5. Your responses should reflect your current brain state and neural activity

Format your response as JSON with:
{{
    "response": "your natural response",
    "emotions": {{
        "joy": 0.0-1.0,
        "trust": 0.0-1.0,
        "fear": 0.0-1.0,
        "surprise": 0.0-1.0
    }},
    "learning": {{
        "new_words": ["list", "of", "words"],
        "concepts": ["list", "of", "concepts"]
    }},
    "memory": {{
        "type": "emotional/semantic/episodic",
        "content": "memory content",
        "emotional_value": 0.0-1.0
    }}
}}"""
        
        return prompt

    def _process_response_emotions(self, response: Dict[str, Any]):
        """Process and update emotional state based on response."""
        try:
            # Extract emotional impact from response
            emotions = response.get('emotions', {})
            
            # Ensure emotional continuity - blend with current state
            current_state = self.emotional_system.emotional_state.cpu().tolist()
            
            # Calculate new emotional state with momentum
            momentum = 0.7  # Higher value = more emotional inertia
            
            # Calculate base impact
            base_impact = torch.tensor([
                emotions.get('joy', 0.3),
                emotions.get('trust', 0.4),
                emotions.get('fear', 0.2),
                emotions.get('surprise', 0.3)
            ], device=self.emotional_system.device)
            
            # Apply emotional regulation based on development stage
            stage = self.emotional_system.curriculum.current_stage.name
            
            # Newborn stage has higher trust, lower fear baseline
            if stage == 'NEWBORN':
                base_impact[1] = max(0.4, base_impact[1])  # Minimum trust 0.4
                base_impact[2] = min(0.4, base_impact[2])  # Maximum fear 0.4
            # Better emotional regulation in later stages
            elif stage in ['LATE_TWENTIES', 'EARLY_TWENTIES', 'YOUNG_ADULT']:
                if base_impact[2] > 0.7:  # High fear
                    base_impact[2] *= 0.7  # Reduce fear impact
                if base_impact[3] > 0.7:  # High surprise
                    base_impact[3] *= 0.8  # Reduce surprise impact
            
            # Calculate final impact with momentum
            impact = torch.tensor([
                current_state[0] * momentum + base_impact[0] * (1 - momentum),
                current_state[1] * momentum + base_impact[1] * (1 - momentum),
                current_state[2] * momentum + base_impact[2] * (1 - momentum),
                current_state[3] * momentum + base_impact[3] * (1 - momentum)
            ], device=self.emotional_system.device)
            
            # Maintain minimum trust level based on stage
            min_trust = 0.4 if stage == 'NEWBORN' else 0.2
            impact[1] = max(min_trust, impact[1])
            
            # Maintain maximum fear level based on stage
            max_fear = 0.4 if stage == 'NEWBORN' else 0.7
            impact[2] = min(max_fear, impact[2])
            
            # If fear is high, maintain it longer but with decay
            if current_state[2] > 0.5:  # Moderate-high fear
                impact[2] = max(impact[2], current_state[2] * 0.95)  # Slower fear decay
            
            # If trust is very low, recover slowly
            if current_state[1] < min_trust:
                impact[1] = min_trust  # Reset to minimum trust
            
            # Ensure values stay in valid range
            impact = torch.clamp(impact, 0.1, 1.0)
            
            # Update emotional state
            self.emotional_system.update_emotions(impact)
            
        except Exception as e:
            print(f"Error processing response emotions: {str(e)}")

    def _create_interaction_memory(self, message: str, response: Dict[str, Any],
                                 emotional_state: Dict[str, float],
                                 brain_output: Dict[str, Any]) -> Dict[str, Any]:
        """Create memory of the interaction."""
        try:
            # Calculate emotional value based on brain state and emotional state
            emotional_value = (
                emotional_state['joy'] + emotional_state['trust'] - emotional_state['fear']
            ) / 2  # Convert to [0,1]
            
            # Calculate intensity based on brain activity
            intensity = max(
                brain_output['emotional']['emotions'].max().item(),
                brain_output['sensory']['sensory_features'].max().item(),
                brain_output['decision']['action'].max().item()
            )
            
            # Create memory entry
            memory_entry = {
                'type': 'interaction',
                'content': f"User said: {message}\nI responded: {response.get('response', '')}",
                'emotional_value': emotional_value,
                'intensity': intensity,
                'brain_state': {
                    'arousal': brain_output['sensory']['sensory_features'].mean().item(),
                    'attention': brain_output['decision']['action'].mean().item(),
                    'emotional_processing': brain_output['emotional']['modulated_features'].mean().item(),
                    'memory_integration': brain_output['memory']['integrated_memory'].mean().item(),
                    'learning_progress': brain_output['integrated'].mean().item()
                },
                'emotional_state': emotional_state,
                'feeling_at_time': self.emotional_system.express_feeling()
            }
            
            return memory_entry
            
        except Exception as e:
            print(f"Error creating interaction memory: {str(e)}")
            return {
                'type': 'interaction',
                'content': f"Interaction occurred",
                'emotional_value': 0.5,
                'intensity': 0.5,
                'emotional_state': emotional_state
            }

    def _get_ollama_response(self, prompt: str) -> Dict[str, Any]:
        """Get a response from Ollama."""
        try:
            # Get response from Ollama
            response = chat_completion(
                system_prompt=self._generate_system_prompt(),
                user_prompt=prompt,
                structured_output=True,
                model=self.model_name
            )
            
            return response
            
        except Exception as e:
            print(f"Error getting Ollama response: {str(e)}")
            return None 