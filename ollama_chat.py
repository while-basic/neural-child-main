# ollama_chat.py
# Description: Chat interface for neural child development using Ollama
# Created by: Christopher Celaya

import requests
import json
import torch
from typing import Dict, Any, Optional, Union
from datetime import datetime
from llm_module import chat_completion

class OllamaChat:
    """Interface for Ollama LLM integration"""
    
    def __init__(self, model: str = "llama3"):
        """Initialize Ollama chat interface"""
        self.model = model
        self.base_url = "http://localhost:11434/api"
        self.context = []
        
    def generate_response(self, 
                         message: str, 
                         emotional_state: Dict[str, float],
                         brain_state: Dict[str, Any],
                         stage: str,
                         age_months: int,
                         image_data: Optional[Dict[str, str]] = None) -> str:
        """Generate response using Ollama with emotional context and optional image"""
        
        # Create system prompt with emotional and developmental context
        system_prompt = f"""You are a neural child AI at developmental stage {stage} (age {age_months} months).
Your current emotional state is:
{json.dumps(emotional_state, indent=2)}

Your brain state shows:
{json.dumps(brain_state, indent=2)}

Respond naturally as a child of this age and emotional state would.
Do not mention specific emotion values or numbers.
Express emotions through tone, word choice, and behavior.
Use *asterisks* to show physical reactions or behaviors.

Remember:
- Stay in character as a child AI
- Express emotions naturally through your response
- Be consistent with your developmental stage
- Show appropriate emotional reactions
- Address the user as Mr. Chris
"""

        try:
            # Prepare the request
            payload = {
                "model": self.model,
                "prompt": message,
                "system": system_prompt,
                "stream": False
            }
            
            # Add image data if provided and using llava model
            if image_data and self.model == "llava":
                payload["images"] = [image_data["data"]]
            
            # Make request to Ollama
            response = requests.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            generated_text = result.get('response', '')
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error generating Ollama response: {str(e)}")
            return "I'm having trouble processing that right now..."
            
    def get_sentiment(self, text: str) -> float:
        """Get sentiment score from Ollama"""
        try:
            system_prompt = """Analyze the sentiment of the following text.
Return only a number between -1.0 (most negative) and 1.0 (most positive).
Do not include any other text or explanation."""

            payload = {
                "model": self.model,
                "prompt": text,
                "system": system_prompt,
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            sentiment_text = result.get('response', '0.0')
            
            try:
                sentiment = float(sentiment_text)
                return max(-1.0, min(1.0, sentiment))
            except ValueError:
                return 0.0
                
        except Exception as e:
            print(f"Error getting sentiment: {str(e)}")
            return 0.0

class OllamaChildChat:
    """Neural child chat system using Ollama"""
    
    def __init__(self, memory_store, emotional_system, language_system, model_name="llama3"):
        """Initialize neural child chat system"""
        self.ollama = OllamaChat(model=model_name)
        self.memory_store = memory_store
        self.emotional_system = emotional_system
        self.language_system = language_system
        self.last_interaction = None
        
    def chat(self, message: str) -> Dict[str, Any]:
        """Process chat message and generate response"""
        try:
            # Update interaction time
            current_time = datetime.now()
            time_since_last = None
            if self.last_interaction:
                time_since_last = (current_time - self.last_interaction).total_seconds()
            self.last_interaction = current_time
            
            # Get current states
            emotional_state = {
                'joy': float(self.emotional_system.emotional_state[0]),
                'trust': float(self.emotional_system.emotional_state[1]),
                'fear': float(self.emotional_system.emotional_state[2]),
                'surprise': float(self.emotional_system.emotional_state[3]),
                'sadness': float(self.emotional_system.emotional_state[4]),
                'anger': float(self.emotional_system.emotional_state[5]),
                'disgust': float(self.emotional_system.emotional_state[6]),
                'anticipation': float(self.emotional_system.emotional_state[7])
            }
            
            brain_state = {
                'arousal': self.emotional_system.brain.brain_state.arousal,
                'attention': self.emotional_system.brain.brain_state.attention,
                'emotional_valence': self.emotional_system.brain.brain_state.emotional_valence,
                'consciousness': self.emotional_system.brain.brain_state.consciousness
            }
            
            # Generate response
            response = self.ollama.generate_response(
                message=message,
                emotional_state=emotional_state,
                brain_state=brain_state,
                stage=self.emotional_system.curriculum.current_stage.name,
                age_months=self.emotional_system.age()
            )
            
            # Get sentiment and update emotional state
            sentiment = self.ollama.get_sentiment(response)
            
            # Create new emotional state tensor with all 8 emotions
            new_emotional_state = torch.tensor([
                max(0.1, emotional_state['joy'] + sentiment * 0.2),
                max(0.1, emotional_state['trust'] + sentiment * 0.1),
                max(0.1, emotional_state['fear'] - sentiment * 0.1),
                emotional_state['surprise'],
                max(0.1, emotional_state['sadness'] - sentiment * 0.15),
                max(0.1, emotional_state['anger'] - sentiment * 0.1),
                emotional_state['disgust'],
                max(0.1, emotional_state['anticipation'] + sentiment * 0.15)
            ], device=self.emotional_system.emotional_state.device)
            
            # Update emotional state
            self.emotional_system.update_emotions(new_emotional_state)
            
            # Process language learning
            learning = self.language_system.process_input(response)
            
            # Create memory
            memory_type = "episodic"
            memory_content = f"User: {message}\nResponse: {response}"
            
            # Create embedding for memory
            embedding = self.memory_store.model.encode(memory_content).tolist()
            
            memory_id = self.memory_store.store_episodic_memory(
                content=memory_content,
                embedding=embedding,
                metadata={
                    'type': memory_type,
                    'sentiment': sentiment,
                    'time_since_last': time_since_last,
                    'stage': self.emotional_system.curriculum.current_stage.name,
                    'age_months': self.emotional_system.age(),
                    **emotional_state  # Include emotional state in metadata
                }
            )
            
            return {
                'response': response,
                'emotions': emotional_state,
                'brain_state': brain_state,
                'learning': learning,
                'memory': {
                    'type': memory_type,
                    'content': memory_content,
                    'emotional_value': (sentiment + 1) / 2  # Convert to 0-1 range
                }
            }
            
        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
            return {
                'response': "I'm having trouble processing that right now...",
                'emotions': emotional_state,
                'brain_state': brain_state
            }

    def process_emotional_input(self, input_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process emotional input and store in memory"""
        try:
            # Get current emotional state
            emotional_state = {
                'joy': float(self.emotional_system.emotional_state[0]),
                'trust': float(self.emotional_system.emotional_state[1]),
                'fear': float(self.emotional_system.emotional_state[2]),
                'surprise': float(self.emotional_system.emotional_state[3]),
                'sadness': float(self.emotional_system.emotional_state[4]),
                'anger': float(self.emotional_system.emotional_state[5]),
                'disgust': float(self.emotional_system.emotional_state[6]),
                'anticipation': float(self.emotional_system.emotional_state[7])
            }
            
            # Calculate valence and arousal
            valence = (emotional_state['joy'] + emotional_state['trust'] + emotional_state['anticipation'] - 
                      emotional_state['fear'] - emotional_state['sadness'] - emotional_state['disgust']) / 3
            arousal = (emotional_state['surprise'] + emotional_state['fear'] + 
                      emotional_state['anger'] + emotional_state['anticipation']) / 4
            
            # Calculate intensity based on emotional extremes
            intensity = max(
                emotional_state['fear'],
                emotional_state['surprise'],
                emotional_state['anger'],
                emotional_state['joy'],
                abs(valence)
            )
            
            # Create memory
            memory_type = "episodic"
            memory_content = f"User: {input_text}\nResponse: {intensity}"
            memory_id = self.memory_store.store_episodic_memory(
                content=memory_content,
                emotional_state=emotional_state,
                metadata={
                    'type': memory_type,
                    'intensity': intensity,
                    'context': context
                }
            )
            
            return {
                'response': intensity,
                'emotions': emotional_state,
                'memory': {
                    'type': memory_type,
                    'content': memory_content,
                    'emotional_value': (intensity + 1) / 2  # Convert to 0-1 range
                }
            }
            
        except Exception as e:
            print(f"Error processing emotional input: {str(e)}")
            return {
                'response': "I'm having trouble processing that right now...",
                'emotions': emotional_state
            } 

def get_child_response(text: str, stage: str, age_months: int) -> str:
    """
    Get response from neural child based on developmental stage
    
    Args:
        text (str): Input text to respond to
        stage (str): Current developmental stage
        age_months (int): Age in months
        
    Returns:
        str: Generated response
    """
    try:
        prompt = f"""
        You are a neural child AI at developmental stage {stage} (age {age_months} months).
        Respond to the following input in a way appropriate for your developmental stage:
        
        Input: {text}
        
        Response:"""
        
        return chat_completion(prompt)
        
    except Exception as e:
        print(f"Error getting child response: {e}")
        return ""

def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict[str, float]: Sentiment scores
    """
    try:
        prompt = f"""
        Analyze the sentiment of the following text.
        Return scores (0-1) for: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
        
        Text: {text}
        
        Format response as JSON:
        {{
            "joy": 0.5,
            "trust": 0.3,
            ...
        }}
        """
        
        response = chat_completion(prompt, structured_output=True)
        return response if response else {}
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {} 