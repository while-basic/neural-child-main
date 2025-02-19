# text_embed.py

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional

model = None

def get_embeddings(text, model_name="all-mpnet-base-v2", device=None):
    """Get embeddings for text using a local SentenceTransformer model.
    
    Args:
        text (str): Text to embed
        model_name (str): Name of the model to use, defaults to all-mpnet-base-v2 (768 dims)
        device (str): Device to use for model, defaults to CUDA if available
        
    Returns:
        dict: Dictionary containing embeddings and metadata
    """
    global model
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    try:
        if model is None:
            print(f"Loading embedding model {model_name} on {device}...")
            model = SentenceTransformer(model_name).to(device)
            
        # Generate embeddings
        embeddings = model.encode(text, convert_to_tensor=True)
        
        # Format response similar to OpenAI API
        return {
            "data": [{
                "embedding": embeddings.cpu().tolist(),
                "index": 0,
                "object": "embedding"
            }],
            "model": model_name,
            "object": "list",
            "usage": {
                "prompt_tokens": len(text.split()),
                "total_tokens": len(text.split())
            }
        }
        
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        # Return zero embedding as fallback
        return {
            "data": [{
                "embedding": [0.0] * (768 if "mpnet" in model_name else 384),
                "index": 0, 
                "object": "embedding"
            }],
            "model": model_name,
            "object": "list",
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }
