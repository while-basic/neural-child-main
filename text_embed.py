# text_embed.py
"""
Text Embedding Module for Neural Child Development System
Created by: Christopher Celaya
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Initialize the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = None
model = None

def initialize_embedding_model():
    """Initialize the embedding model and tokenizer"""
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

def get_embeddings(text, normalize=True):
    """Get embeddings for input text"""
    try:
        # Initialize if not already done
        if tokenizer is None or model is None:
            initialize_embedding_model()
            
        # Tokenize and prepare input
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Use mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        # Convert to numpy and return in expected format
        embedding_np = embeddings.cpu().numpy()
        return {
            'data': [{
                'embedding': embedding_np[0].tolist()
            }]
        }
        
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        # Return zero embedding as fallback
        return {
            'data': [{
                'embedding': np.zeros(384).tolist()  # Default embedding size for MiniLM
            }]
        }
