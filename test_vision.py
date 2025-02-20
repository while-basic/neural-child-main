# test_vision.py
# Description: Test script for the neural child vision development system
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from digital_child import DigitalChild
from logger import DevelopmentLogger
from development_stages import DevelopmentStage
import requests
from io import BytesIO
import json
from typing import Dict, List, Any
import os

def call_ollama_vision(image_path: str, prompt: str) -> Dict[str, Any]:
    """Call Ollama's vision model for image analysis"""
    try:
        # Convert image to base64
        with open(image_path, 'rb') as image_file:
            import base64
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare the request
        payload = {
            "model": "llava",
            "prompt": prompt,
            "images": [image_data],
            "stream": False
        }
        
        # Make the request to Ollama
        response = requests.post('http://localhost:11434/api/generate', json=payload)
        return response.json()
        
    except Exception as e:
        print(f"Error calling Ollama: {str(e)}")
        return {'error': str(e)}

def test_vision_development():
    """Test the vision development system with various images"""
    print("\n=== Testing Neural Child Vision Development ===\n")
    
    # Initialize the system
    logger = DevelopmentLogger()
    child = DigitalChild(logger)
    
    # Test images for different developmental concepts
    test_images = {
        'faces': 'test_images/faces.jpg',
        'objects': 'test_images/objects.jpg',
        'patterns': 'test_images/patterns.jpg',
        'scenes': 'test_images/scenes.jpg',
        'symbols': 'test_images/symbols.jpg'
    }
    
    # Create test_images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Test through different developmental stages
    stages = [
        DevelopmentStage.NEWBORN,
        DevelopmentStage.INFANT,
        DevelopmentStage.EARLY_TODDLER,
        DevelopmentStage.LATE_TODDLER,
        DevelopmentStage.EARLY_PRESCHOOL,
        DevelopmentStage.LATE_PRESCHOOL,
        DevelopmentStage.EARLY_CHILDHOOD
    ]
    
    for stage in stages:
        print(f"\nTesting Stage: {stage.name}")
        child.set_stage(stage)
        
        for image_type, image_path in test_images.items():
            print(f"\nProcessing {image_type} image...")
            
            try:
                # Load and process image
                image = Image.open(image_path)
                image_tensor = transforms.ToTensor()(image).unsqueeze(0)
                
                # Process with our vision development system
                vision_result = child.process_visual_input(image_tensor)
                
                print("Vision Development Results:")
                print(f"Stage: {vision_result['stage']}")
                print("Metrics:", json.dumps(vision_result['metrics'], indent=2))
                
                # Process with Ollama for enhanced understanding
                stage_appropriate_prompt = get_stage_appropriate_prompt(stage, image_type)
                ollama_result = call_ollama_vision(image_path, stage_appropriate_prompt)
                
                print("\nOllama Vision Analysis:")
                if 'error' not in ollama_result:
                    print(ollama_result['response'])
                else:
                    print(f"Ollama Error: {ollama_result['error']}")
                
                # Compare and integrate results
                integrated_analysis = integrate_vision_results(
                    vision_result,
                    ollama_result,
                    stage
                )
                
                print("\nIntegrated Analysis:")
                print(json.dumps(integrated_analysis, indent=2))
                
            except FileNotFoundError:
                print(f"Warning: Test image {image_path} not found. Skipping...")
            except Exception as e:
                print(f"Error processing {image_type}: {str(e)}")
                
        print("\nStage Development Summary:")
        status = child.get_development_status()
        print(json.dumps(status['vision_development'], indent=2))
        print("="*50)

def get_stage_appropriate_prompt(stage: DevelopmentStage, image_type: str) -> str:
    """Generate stage-appropriate prompts for Ollama vision analysis"""
    prompts = {
        DevelopmentStage.NEWBORN: {
            'faces': "Describe the basic shapes and contrasts in this image, focusing on face-like patterns.",
            'objects': "Identify any large, high-contrast shapes or patterns.",
            'patterns': "Describe the basic light and dark patterns.",
            'scenes': "Identify the main areas of light and shadow.",
            'symbols': "Describe the basic visual elements in simple terms."
        },
        DevelopmentStage.INFANT: {
            'faces': "Identify and describe any faces, especially focusing on eyes and mouths.",
            'objects': "Point out any bright, colorful objects.",
            'patterns': "Describe the colors and basic patterns.",
            'scenes': "Identify main colors and large objects.",
            'symbols': "Describe any simple, recognizable shapes."
        },
        DevelopmentStage.EARLY_TODDLER: {
            'faces': "Describe facial expressions and basic emotions.",
            'objects': "Name common objects and their basic features.",
            'patterns': "Identify repeating patterns and shapes.",
            'scenes': "Describe the main elements in the scene.",
            'symbols': "Identify basic symbols and shapes."
        },
        DevelopmentStage.LATE_TODDLER: {
            'faces': "Describe people's features and expressions in detail.",
            'objects': "Describe objects, their properties, and basic relationships.",
            'patterns': "Explain patterns and their relationships.",
            'scenes': "Describe the scene and relationships between elements.",
            'symbols': "Identify and explain simple symbols."
        },
        DevelopmentStage.EARLY_PRESCHOOL: {
            'faces': "Analyze facial expressions, emotions, and social cues.",
            'objects': "Describe objects, their functions, and relationships.",
            'patterns': "Explain complex patterns and their organization.",
            'scenes': "Describe the scene, context, and story elements.",
            'symbols': "Identify symbols and their basic meanings."
        }
    }
    
    # Default to most advanced prompt if stage not found
    stage_prompts = prompts.get(stage, prompts[DevelopmentStage.EARLY_PRESCHOOL])
    return stage_prompts.get(image_type, "Describe what you see in this image.")

def integrate_vision_results(
    vision_result: Dict,
    ollama_result: Dict,
    stage: DevelopmentStage
) -> Dict[str, Any]:
    """Integrate results from vision development system and Ollama"""
    
    # Extract relevant metrics and features
    metrics = vision_result.get('metrics', {})
    features = vision_result.get('features', {})
    
    # Get Ollama's analysis
    ollama_analysis = ollama_result.get('response', '')
    
    # Create integrated analysis based on developmental stage
    integrated = {
        'stage': stage.name,
        'development_metrics': {
            'visual_acuity': metrics.get('visual_acuity', 0),
            'pattern_recognition': metrics.get('pattern_recognition', 0),
            'object_recognition': features.get('object_recognition', 0)
        },
        'visual_understanding': {
            'system_features': features.get('feature_type', 'unknown'),
            'ollama_analysis': ollama_analysis
        },
        'stage_appropriate_capabilities': {}
    }
    
    # Add stage-specific capabilities
    if stage == DevelopmentStage.NEWBORN:
        integrated['stage_appropriate_capabilities'] = {
            'contrast_sensitivity': features.get('contrast_detection', 0),
            'light_detection': features.get('light_sensitivity', 0)
        }
    elif stage == DevelopmentStage.INFANT:
        integrated['stage_appropriate_capabilities'] = {
            'face_detection': features.get('face_detection', 0),
            'color_sensitivity': features.get('color_sensitivity', 0)
        }
    elif stage == DevelopmentStage.EARLY_TODDLER:
        integrated['stage_appropriate_capabilities'] = {
            'object_recognition': features.get('object_recognition', 0),
            'depth_perception': metrics.get('depth_perception', 0)
        }
    elif stage == DevelopmentStage.LATE_TODDLER:
        integrated['stage_appropriate_capabilities'] = {
            'detail_perception': metrics.get('visual_acuity', 0),
            'object_permanence': metrics.get('object_permanence', 0)
        }
    elif stage == DevelopmentStage.EARLY_PRESCHOOL:
        integrated['stage_appropriate_capabilities'] = {
            'scene_understanding': features.get('scene_understanding', 0),
            'symbol_recognition': features.get('symbol_recognition', 0)
        }
    
    return integrated

if __name__ == "__main__":
    test_vision_development() 