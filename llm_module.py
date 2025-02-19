# llm_module.py

import requests
import json
from typing import Optional, Dict, Any
from schemas import MotherResponse

def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "qwen2.5-7b-instruct",
    temperature: float = 0.7,
    max_tokens: int = -1,
    stream: bool = False,  # Default to non-streaming for better reliability
    server_url: str = "http://localhost:1234/v1/chat/completions",
    structured_output: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Enhanced with automatic structured response handling and better error handling
    """
    # Generate JSON schema if structured output requested
    response_schema = None
    if structured_output:
        response_schema = {
            "name": MotherResponse.__name__,
            "strict": True,
            "schema": MotherResponse.model_json_schema()
        }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }

    if response_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": response_schema
        }

    headers = {"Content-Type": "application/json"}

    try:
        # Reduced timeout and non-streaming for initial test
        response = requests.post(server_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data or "choices" not in data or not data["choices"]:
            print("Invalid response format from LLM server")
            return None
            
        raw_content = data["choices"][0].get("message", {}).get("content", "")
        
        if not raw_content:
            print("Empty response from LLM server")
            return None

        if response_schema:
            try:
                return MotherResponse(**json.loads(raw_content)).dict()
            except json.JSONDecodeError as e:
                print(f"Failed to parse structured LLM response: {e}")
                return None
            except Exception as e:
                print(f"Error processing structured response: {e}")
                return None
                
        return raw_content

    except requests.exceptions.Timeout:
        print("Request timed out. Please check if LM Studio server is responsive.")
    except requests.exceptions.ConnectionError:
        print("Connection error. Please check if LM Studio server is running.")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None
