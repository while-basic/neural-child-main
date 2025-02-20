# llm_module.py
# Description: LLM interaction module for neural child development
# Created by: Christopher Celaya

import requests
import json
import time
import re
from typing import Optional, Dict, Any
from schemas import MotherResponse

DEFAULT_RESPONSE = {
    "content": "",
    "joy": 0.5,
    "trust": 0.5,
    "fear": 0.2,
    "surprise": 0.3,
    "reward_score": 0.5,
    "success_metric": 0.5,
    "complexity_rating": 0.3,
    "self_critique_score": 0.5
}

def format_prompt_for_json(schema: Dict) -> str:
    """Format the prompt to guide the model to generate proper JSON."""
    return """
    IMPORTANT: You must respond with a valid JSON object and nothing else. No explanations or additional text.
    The JSON object must contain these exact fields with values in the specified ranges:

    {
        "response": "your natural response",
        "emotions": {
            "joy": (number between 0 and 1),
            "trust": (number between 0 and 1),
            "fear": (number between 0 and 1),
            "surprise": (number between 0 and 1)
        },
        "learning": {
            "new_words": ["list", "of", "words"],
            "concepts": ["list", "of", "concepts"]
        },
        "memory": {
            "type": "emotional/semantic/episodic",
            "content": "memory content",
            "emotional_value": (number between 0 and 1)
        }
    }

    Example valid response:
    {
        "response": "I feel happy when you talk to me!",
        "emotions": {
            "joy": 0.7,
            "trust": 0.8,
            "fear": 0.2,
            "surprise": 0.3
        },
        "learning": {
            "new_words": ["happy", "talk"],
            "concepts": ["emotions", "communication"]
        },
        "memory": {
            "type": "emotional",
            "content": "Felt joy during conversation",
            "emotional_value": 0.7
        }
    }

    Remember:
    1. Only output the JSON object, nothing else
    2. All numerical values must be between 0 and 1
    3. Include all required fields
    4. Use proper JSON format with double quotes
    """

def chat_completion(system_prompt: str, user_prompt: str, structured_output: bool = False, model: str = "artifish/llama3.2-uncensored") -> Optional[Dict]:
    """Get a chat completion from the Ollama API with improved error handling"""
    try:
        # Format the prompt to encourage proper JSON formatting
        if structured_output:
            json_format_prompt = format_prompt_for_json({})
            system_prompt = f"{system_prompt}\n\n{json_format_prompt}"
            user_prompt = f"{user_prompt}\nRespond with a complete JSON object following the specified format."
        
        # Prepare the request
        url = "http://localhost:11434/api/chat"
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "format": "json" if structured_output else None
        }
        
        # Make the request with retry logic
        max_retries = 3
        timeout = 30  # 30 seconds timeout
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, timeout=timeout)
                response.raise_for_status()
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract content from Ollama's response format
                    content = result.get('response', {}).get('content', '')
                    if not content:
                        content = result.get('message', {}).get('content', '')
                    
                    if structured_output:
                        try:
                            # Find and clean the JSON part of the response
                            json_start = content.find('{')
                            json_end = content.rfind('}') + 1
                            
                            if json_start != -1 and json_end != -1:
                                json_str = content[json_start:json_end]
                                
                                # Clean the JSON string
                                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                                json_str = re.sub(r'\s+', ' ', json_str)
                                json_str = re.sub(r',\s*}', '}', json_str)
                                json_str = re.sub(r',\s*]', ']', json_str)
                                
                                try:
                                    # Parse the JSON
                                    response_data = json.loads(json_str)
                                    
                                    # Ensure response has required fields
                                    if "response" not in response_data:
                                        response_data["response"] = content or "I'm learning to understand that."
                                    
                                    if "emotions" not in response_data:
                                        response_data["emotions"] = {
                                            "joy": 0.3,
                                            "trust": 0.4,
                                            "fear": 0.3,
                                            "surprise": 0.2
                                        }
                                    
                                    if "learning" not in response_data:
                                        response_data["learning"] = {
                                            "new_words": [],
                                            "concepts": []
                                        }
                                    
                                    if "memory" not in response_data:
                                        response_data["memory"] = {
                                            "type": "episodic",
                                            "content": content or "Had an interaction",
                                            "emotional_value": 0.5
                                        }
                                    
                                    # Ensure all numerical values are within bounds
                                    for emotion, value in response_data["emotions"].items():
                                        response_data["emotions"][emotion] = max(0.0, min(1.0, float(value)))
                                    
                                    response_data["memory"]["emotional_value"] = max(0.0, min(1.0, float(response_data["memory"]["emotional_value"])))
                                    
                                    return response_data
                                    
                                except json.JSONDecodeError:
                                    print(f"\n❌ Error parsing JSON response: {json_str}")
                                    if attempt == max_retries - 1:
                                        return None
                                    time.sleep(2)  # Wait before retrying
                                    continue
                            else:
                                print("\n❌ No valid JSON found in response")
                                if attempt == max_retries - 1:
                                    return None
                                time.sleep(2)  # Wait before retrying
                                continue
                        except Exception as e:
                            print(f"\n❌ Error processing response: {str(e)}")
                            if attempt == max_retries - 1:
                                return None
                            time.sleep(2)  # Wait before retrying
                            continue
                    else:
                        return content
                break
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Connection error: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)  # Wait before retrying
            
    except Exception as e:
        print(f"\n❌ Error in chat_completion: {str(e)}")
        return None
