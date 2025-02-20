# ollama_chat.py
# Description: Advanced chat system for neural child development using Ollama
# Created by: Christopher Celaya

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer
from llm_module import chat_completion
from memory_store import MemoryStore
from language_development import LanguageDevelopment

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
        self.context_window = 5  # Number of previous messages to maintain context
        
    def _generate_system_prompt(self) -> str:
        """Generate the system prompt based on current development state."""
        # Get emotional state values
        joy = float(self.emotional_system.emotional_state[0])
        trust = float(self.emotional_system.emotional_state[1])
        fear = float(self.emotional_system.emotional_state[2])
        surprise = float(self.emotional_system.emotional_state[3])
        
        current_feeling = self.emotional_system.express_feeling()
        
        return f"""You are a neural child in development. Your responses should reflect:
        - Current emotional state: {current_feeling}
          * Joy: {joy:.2f}
          * Trust: {trust:.2f}
          * Fear: {fear:.2f}
          * Surprise: {surprise:.2f}
        - Language development stage: {self.language_system.get_current_stage()}
        - Learning capabilities: Actively learning new words and concepts
        - Memory formation: Creating memories with emotional context
        
        Respond naturally as a developing child would, showing:
        1. Appropriate emotional responses
        2. Age-appropriate language use
        3. Curiosity and desire to learn
        4. Formation of memories and attachments
        
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
        query_embedding = self.sentence_encoder.encode(message).tolist()
        return self.memory_store.query_memories(query_embedding, n_results=limit)

    def chat(self, message: str) -> Dict:
        """Process a chat message and return a response with learning and emotional updates."""
        try:
            # Get relevant memories for context
            relevant_memories = self._get_relevant_memories(message)
            
            # Extract learnable content
            new_words, new_concepts = self._extract_learnable_content(message)
            
            # Build context from conversation history and memories
            context = "\n".join([
                f"Previous memory: {mem['content']}" for mem in relevant_memories
            ])
            
            # Generate system prompt
            system_prompt = self._generate_system_prompt()
            
            # Prepare user prompt with context
            user_prompt = f"""Context from memories:
            {context}
            
            Current message: {message}
            
            Respond naturally while learning from the interaction."""
            
            # Get response from Ollama
            response = chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                structured_output=True,
                model=self.model_name
            )
            
            # If response is None or invalid, use default response
            if not response or not isinstance(response, dict):
                response = {
                    "response": "I'm learning to understand that.",
                    "emotions": {
                        "joy": 0.3,
                        "trust": 0.4,
                        "fear": 0.3,
                        "surprise": 0.2
                    },
                    "learning": {
                        "new_words": new_words,
                        "concepts": []
                    },
                    "memory": {
                        "type": "episodic",
                        "content": f"Someone said: {message}",
                        "emotional_value": 0.5
                    }
                }
            
            # Ensure response has all required fields
            if "response" not in response:
                response["response"] = response.get("content", "I'm learning to understand that.")
            
            if "emotions" not in response:
                response["emotions"] = {
                    "joy": 0.3,
                    "trust": 0.4,
                    "fear": 0.3,
                    "surprise": 0.2
                }
            
            if "learning" not in response:
                response["learning"] = {
                    "new_words": new_words,
                    "concepts": []
                }
            
            if "memory" not in response:
                response["memory"] = {
                    "type": "episodic",
                    "content": f"Someone said: {message}",
                    "emotional_value": 0.5
                }
            
            # Update emotional state
            self._update_emotional_state(response.get("emotions", {}))
            
            # Process learning outcomes
            learning = response.get("learning", {})
            for word in learning.get("new_words", []):
                self.language_system.learn_word(word)
            for concept in learning.get("concepts", []):
                self.language_system.learn_concept(concept)
            
            # Create memory if specified
            memory_data = response.get("memory", {})
            if memory_data:
                self._create_memory(
                    content=memory_data["content"],
                    memory_type=memory_data["type"],
                    emotional_value=memory_data["emotional_value"]
                )
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response["response"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Trim conversation history to maintain context window
            if len(self.conversation_history) > self.context_window * 2:
                self.conversation_history = self.conversation_history[-self.context_window * 2:]
            
            return response
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            # Return a safe default response
            return {
                "response": "I'm having trouble processing that right now.",
                "emotions": {
                    "joy": 0.3,
                    "trust": 0.4,
                    "fear": 0.3,
                    "surprise": 0.2
                },
                "learning": {
                    "new_words": [],
                    "concepts": []
                },
                "memory": {
                    "type": "error",
                    "content": "Had trouble processing an interaction",
                    "emotional_value": 0.3
                }
            } 