# llm_module.py

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
        "content": "your response message as a string",
        "joy": (number between 0 and 1),
        "trust": (number between 0 and 1),
        "fear": (number between 0 and 1),
        "surprise": (number between 0 and 1),
        "reward_score": (number between 0 and 1),
        "success_metric": (number between 0 and 1),
        "complexity_rating": (number between 0 and 1),
        "self_critique_score": (number between 0 and 1)
    }

    Example valid response:
    {
        "content": "Hello, I am your mother. Let's play together! [SMILE]",
        "joy": 0.7,
        "trust": 0.8,
        "fear": 0.2,
        "surprise": 0.3,
        "reward_score": 0.6,
        "success_metric": 0.5,
        "complexity_rating": 0.4,
        "self_critique_score": 0.5
    }

    Remember:
    1. Only output the JSON object, nothing else
    2. All numerical values must be between 0 and 1
    3. Include all required fields
    4. Use proper JSON format with double quotes
    """

def chat_completion(system_prompt, user_prompt, structured_output=False, model="llama2"):
    """Get a chat completion from the Ollama API with improved emotional context handling"""
    try:
        # Format the prompt to encourage proper JSON formatting
        if structured_output:
            json_format_prompt = """
            IMPORTANT: Respond with a valid JSON object containing these fields:
            {
                "content": "your response message",
                "emotion": "primary emotional state",
                "intention": "your purpose or goal",
                "developmental_focus": "area of development",
                "joy": (0.0-1.0),
                "trust": (0.0-1.0),
                "fear": (0.0-1.0),
                "surprise": (0.0-1.0),
                "reward_score": (0.0-1.0),
                "success_metric": (0.0-1.0),
                "complexity_rating": (0.0-1.0),
                "self_critique_score": (0.0-1.0)
            }
            
            Guidelines:
            1. All numerical values must be between 0.0 and 1.0
            2. Emotion should match the emotional values
            3. Content should be natural and emotionally appropriate
            4. Include all fields with appropriate values
            """
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
            "stream": False
        }
        
        # Make the request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Request error: {str(e)}")
                    return None
                time.sleep(1)  # Wait before retrying
                
        # Parse the response
        if response.status_code == 200:
            result = response.json()
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
                        
                        # Parse the JSON
                        response_data = json.loads(json_str)
                        
                        # Ensure all numerical values are within bounds
                        for key, value in response_data.items():
                            if isinstance(value, (int, float)):
                                response_data[key] = max(0.0, min(1.0, float(value)))
                        
                        return response_data
                    else:
                        print("No valid JSON found in response")
                        return None
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    return None
            else:
                return content
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error in chat_completion: {str(e)}")
        return None
