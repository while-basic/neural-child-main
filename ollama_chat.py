# ollama_chat.py
# Description: Integration with Ollama LLM for neural child responses
# Created by: Christopher Celaya

import requests
import json
from typing import Dict, Any, Optional

class OllamaChat:
    """Interface for Ollama LLM integration"""
    
    def __init__(self, model: str = "llama2"):
        """Initialize Ollama chat interface"""
        self.model = model
        self.base_url = "http://localhost:11434/api"
        self.context = []
        
    def generate_response(self, 
                         message: str, 
                         emotional_state: Dict[str, float],
                         brain_state: Dict[str, Any],
                         stage: str,
                         age_months: int) -> str:
        """Generate response using Ollama with emotional context"""
        
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