# llm_module.py
# Description: LLM integration for neural child development
# Created by: Christopher Celaya

import requests
from typing import Dict, Any, Optional, Union
import json

def chat_completion(prompt: str, structured_output: bool = False) -> Union[str, Dict[str, Any]]:
    """
    Get completion from local Ollama instance
    
    Args:
        prompt (str): The prompt to send to the model
        structured_output (bool): Whether to parse response as JSON
        
    Returns:
        Union[str, Dict[str, Any]]: Model response as string or parsed JSON
    """
    try:
        # Call local Ollama instance
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama2',
                'prompt': prompt,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '')
            
            if structured_output:
                try:
                    # Try to parse as JSON
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {}
            
            return text
            
        return "" if structured_output else "I apologize, but my developer sucks and needs to fix this."
        
    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        return {} if structured_output else "I apologize, but I'm having trouble connecting to my language model."
