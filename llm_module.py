# llm_module.py

import requests
import json
import time
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
    """
    Get a chat completion from the Ollama API
    """
    try:
        # Format the prompt to encourage proper JSON formatting
        if structured_output:
            system_prompt = system_prompt + "\nYou must respond with a complete, valid JSON object."
            user_prompt = user_prompt + "\nRespond ONLY with a valid JSON object, nothing else."
        
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
                    # Find the JSON part of the response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = content[start_idx:end_idx]
                        
                        # Clean the JSON string
                        json_str = json_str.replace('\n', ' ')
                        json_str = json_str.replace('\r', ' ')
                        json_str = json_str.replace('\\', '\\\\')
                        json_str = json_str.replace('\\"', '"')
                        json_str = json_str.replace(',"', ', "')
                        json_str = json_str.replace('"}', '" }')
                        json_str = json_str.replace('"{', '{ "')
                        
                        # Parse the JSON
                        return json.loads(json_str)
                    else:
                        print("No valid JSON found in response")
                        print(f"Raw content: {content}")
                        return None
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON structure: {str(e)}")
                    print(f"Attempted to parse: {json_str}")
                    return None
            else:
                return content
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error in chat_completion: {str(e)}")
        return None
