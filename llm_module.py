# llm_module.py

import requests
import json
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

def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "artifish/llama3.2-uncensored",  # DO NOT CHANGE THIS MODEL
    temperature: float = 0.7,
    max_tokens: int = -1,
    stream: bool = False,
    server_url: str = "http://localhost:11434/api/chat",
    structured_output: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Enhanced with automatic structured response handling and Ollama API support
    """
    try:
        # Generate JSON schema if structured output requested
        if structured_output:
            json_instructions = format_prompt_for_json(MotherResponse.model_json_schema())
            system_prompt = f"{system_prompt}\n\n{json_instructions}"

        # Format messages for Ollama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Prepare Ollama-specific payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens if max_tokens > 0 else 100
            }
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(server_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data or "message" not in data:
            print("Invalid response format from Ollama server")
            return DEFAULT_RESPONSE if structured_output else None
            
        raw_content = data["message"]["content"]
        
        if not raw_content:
            print("Empty response from Ollama server")
            return DEFAULT_RESPONSE if structured_output else None

        if structured_output:
            try:
                # Try to extract JSON from the response
                json_start = raw_content.find('{')
                json_end = raw_content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = raw_content[json_start:json_end]
                    # Clean up common JSON formatting issues
                    json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                    json_str = json_str.replace('None', 'null')  # Replace Python None with JSON null
                    json_str = json_str.replace('True', 'true').replace('False', 'false')  # Fix booleans
                    
                    try:
                        parsed_json = json.loads(json_str)
                        # Ensure all required fields are present with defaults
                        for key, value in DEFAULT_RESPONSE.items():
                            if key not in parsed_json or not isinstance(parsed_json[key], (str, float, int)):
                                parsed_json[key] = value
                                print(f"Missing or invalid field '{key}', using default value: {value}")
                        
                        # Ensure numerical values are within bounds
                        for key, value in parsed_json.items():
                            if key != 'content' and isinstance(value, (float, int)):
                                old_value = value
                                parsed_json[key] = max(0.0, min(1.0, float(value)))
                                if old_value != parsed_json[key]:
                                    print(f"Clamped {key} from {old_value} to {parsed_json[key]}")
                        
                        return parsed_json
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON structure: {str(e)}")
                        print(f"Attempted to parse: {json_str}")
                        return DEFAULT_RESPONSE
                else:
                    print("No valid JSON found in response")
                    print(f"Raw content: {raw_content}")
                    return DEFAULT_RESPONSE
            except Exception as e:
                print(f"Error processing structured response: {str(e)}")
                return DEFAULT_RESPONSE
                
        return raw_content

    except requests.exceptions.Timeout:
        print("Request timed out. Please check if Ollama server is responsive.")
    except requests.exceptions.ConnectionError:
        print("Connection error. Please check if Ollama server is running.")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return DEFAULT_RESPONSE if structured_output else None
