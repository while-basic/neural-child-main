# test_vision.py
# Description: Test script for vision development system
# Created by: Christopher Celaya

import os
import json
import requests
from PIL import Image
import torch
import torchvision.transforms as transforms
from vision_development import VisionDevelopment, DevelopmentStage
import base64

def call_ollama_vision(image_path: str, prompt: str) -> str:
    """Call Ollama's vision model to analyze an image"""
    try:
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Prepare the payload
        payload = {
            "model": "llava",
            "prompt": prompt,
            "images": [image_data],
            "stream": False
        }
        
        # Make the API call
        response = requests.post('http://localhost:11434/api/generate', json=payload)
        if response.status_code == 200:
            # Parse the response line by line
            lines = response.text.strip().split('\n')
            responses = []
            for line in lines:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        responses.append(data['response'])
                except json.JSONDecodeError:
                    continue
            return ' '.join(responses)
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def test_vision_development():
    """
    Test the vision development system with a sample image.
    """
    print("\n=== Neural Child Vision Development Test ===\n")
    
    # Initialize vision development system
    vision = VisionDevelopment()
    
    # Test image path
    image_path = "test_images/faces.jpg"
    print(f"Testing image: {image_path}\n")
    
    # Test each developmental stage
    for stage in DevelopmentStage:
        print(f"Stage: {stage.name}")
        print("=" * 50 + "\n")
        
        try:
            # Process image for current stage
            metrics = vision.process_image(image_path, stage)
            
            # Print metrics in a clean format
            print("Developmental Metrics:")
            print("-" * 20)
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"{key.replace('_', ' ').title():18}: {value:.2f}")
                    else:
                        print(f"{key.replace('_', ' ').title():18}: {value}")
            print("\n")
            
            # Get Ollama's analysis
            try:
                ollama_result = call_ollama_vision(image_path, "Describe what you see in this image in detail:")
                
                # Print child's response based on developmental stage
                print("Child's Response:")
                print("-" * 20)
                if stage == DevelopmentStage.NEWBORN:
                    print("*Cooing sounds*\n")
                elif stage == DevelopmentStage.INFANT:
                    print("*Points and babbles*\n")
                elif stage == DevelopmentStage.EARLY_TODDLER:
                    print("Person!\n")
                elif stage == DevelopmentStage.LATE_TODDLER:
                    print("Navy person\n")
                else:
                    # For older stages, print Ollama's analysis
                    print("Visual Understanding:")
                    print("-" * 20)
                    print(ollama_result + "\n")
                
            except Exception as e:
                print(f"Error calling Ollama: {str(e)}\n")
                print("Child's Response:")
                print("-" * 20)
                print("No response\n")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}\n")
            print("Developmental Metrics:")
            print("-" * 20 + "\n")
            print("Child's Response:")
            print("-" * 20)
            print("No response\n")
        
        print("-" * 50 + "\n")

if __name__ == "__main__":
    test_vision_development() 