# vision_development.py
# Description: Vision development and perception system for the neural child
# Created by: Christopher Celaya

import torch
import torch.nn as nn
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from datetime import datetime

class DevelopmentStage(Enum):
    NEWBORN = 0         # 0-3 months
    INFANT = 1          # 3-6 months
    EARLY_TODDLER = 2   # 6-12 months
    LATE_TODDLER = 3    # 12-18 months
    EARLY_PRESCHOOL = 4 # 18-24 months
    LATE_PRESCHOOL = 5  # 2-3 years
    EARLY_CHILDHOOD = 6 # 3-4 years
    MIDDLE_CHILDHOOD = 7 # 4-5 years
    LATE_CHILDHOOD = 8  # 5-6 years
    PRE_ADOLESCENT = 9  # 6-12 years
    EARLY_TEEN = 10     # 12-14 years
    MID_TEEN = 11      # 14-16 years
    LATE_TEEN = 12     # 16-18 years
    YOUNG_ADULT = 13    # 18-21 years
    EARLY_TWENTIES = 14 # 21-25 years
    LATE_TWENTIES = 15  # 25-30 years

class VisionDevelopment(nn.Module):
    def __init__(self, device='cpu'):
        """Initialize vision development system"""
        super().__init__()
        self.device = device
        self.current_stage = DevelopmentStage.NEWBORN
        self.stage_progress = 0.0
        
        # Initialize vision models
        self.object_recognition = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        self.object_recognition.eval()
        
        # Vision processing layers
        self.visual_attention = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ).to(device)
        
        # Face detection focus (especially important for early stages)
        self.face_detection_focus = 0.8  # High initial focus on faces
        
        # Initialize stage-specific processing
        self.stage_processors = {
            DevelopmentStage.NEWBORN: self._process_newborn_vision,
            DevelopmentStage.INFANT: self._process_infant_vision,
            DevelopmentStage.EARLY_TODDLER: self._process_early_toddler_vision,
            DevelopmentStage.LATE_TODDLER: self._process_late_toddler_vision,
            DevelopmentStage.EARLY_PRESCHOOL: self._process_preschool_vision,
            DevelopmentStage.LATE_PRESCHOOL: self._process_preschool_vision,
            DevelopmentStage.EARLY_CHILDHOOD: self._process_school_age_vision,
            DevelopmentStage.MIDDLE_CHILDHOOD: self._process_school_age_vision,
            DevelopmentStage.LATE_CHILDHOOD: self._process_school_age_vision,
            DevelopmentStage.PRE_ADOLESCENT: self._process_adolescent_vision,
            DevelopmentStage.EARLY_TEEN: self._process_adolescent_vision,
            DevelopmentStage.MID_TEEN: self._process_adolescent_vision,
            DevelopmentStage.LATE_TEEN: self._process_adolescent_vision,
            DevelopmentStage.YOUNG_ADULT: self._process_adult_vision,
            DevelopmentStage.EARLY_TWENTIES: self._process_adult_vision,
            DevelopmentStage.LATE_TWENTIES: self._process_adult_vision
        }
        
        # Vision metrics
        self.metrics = {
            'visual_acuity': 0.1,        # Starts low, improves with development
            'color_perception': 0.0,     # Develops in infant stage
            'depth_perception': 0.0,     # Develops in early toddler stage
            'pattern_recognition': 0.0,  # Gradually improves
            'object_permanence': 0.0,    # Develops in late toddler stage
            'visual_memory': 0.0,        # Gradually improves
            'attention_span': 0.1        # Starts low, improves with development
        }
        
        # Initialize image transforms
        self._initialize_transforms()
        
    def _initialize_transforms(self):
        """Initialize stage-specific image transformations"""
        self.transforms = {
            DevelopmentStage.NEWBORN: transforms.Compose([
                transforms.Grayscale(3),  # Convert to grayscale but keep 3 channels
                transforms.GaussianBlur(kernel_size=15),  # Very blurry vision
                transforms.ColorJitter(brightness=0.5)  # Limited contrast sensitivity
            ]),
            DevelopmentStage.INFANT: transforms.Compose([
                transforms.GaussianBlur(kernel_size=7),  # Less blurry
                transforms.ColorJitter(brightness=0.3, saturation=0.5)  # Developing color vision
            ]),
            DevelopmentStage.EARLY_TODDLER: transforms.Compose([
                transforms.GaussianBlur(kernel_size=3),  # Clearer vision
                transforms.ColorJitter(brightness=0.2, saturation=0.7)  # Better color perception
            ]),
            DevelopmentStage.LATE_TODDLER: transforms.Compose([
                transforms.ColorJitter(brightness=0.1, saturation=0.8)  # Near-normal color vision
            ]),
            # Later stages use minimal or no transformation
            DevelopmentStage.EARLY_PRESCHOOL: transforms.Compose([
                transforms.ColorJitter(saturation=0.1)  # Slight adjustment
            ]),
            DevelopmentStage.LATE_PRESCHOOL: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.EARLY_CHILDHOOD: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.MIDDLE_CHILDHOOD: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.LATE_CHILDHOOD: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.PRE_ADOLESCENT: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.EARLY_TEEN: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.MID_TEEN: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.LATE_TEEN: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.YOUNG_ADULT: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.EARLY_TWENTIES: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ]),
            DevelopmentStage.LATE_TWENTIES: transforms.Compose([
                transforms.Lambda(lambda x: x)  # No transformation
            ])
        }
        
    def process_image(self, image_path: str, stage: DevelopmentStage) -> Dict[str, float]:
        """Process an image for a specific developmental stage"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Apply stage-specific transformations
            if stage in self.transforms:
                image_tensor = self.transforms[stage](image_tensor)
            
            # Calculate metrics based on stage
            metrics = self._calculate_metrics(stage)
            
            return metrics
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return {}
            
    def _calculate_metrics(self, stage: DevelopmentStage) -> Dict[str, float]:
        """Calculate vision metrics based on developmental stage"""
        base_metrics = {
            'visual_acuity': 0.1 + (stage.value * 0.01),
            'color_perception': max(0.0, -0.05 + (stage.value * 0.05)),
            'depth_perception': max(0.0, -0.1 + (stage.value * 0.05)),
            'pattern_recognition': max(0.0, -0.15 + (stage.value * 0.035)),
            'object_permanence': max(0.0, -0.2 + (stage.value * 0.05)),
            'visual_memory': max(0.0, -0.25 + (stage.value * 0.035)),
            'attention_span': 0.1 + (stage.value * 0.01)
        }
        return base_metrics

    def process_visual_input(self, image: torch.Tensor) -> Dict[str, Any]:
        """Process visual input based on current developmental stage"""
        try:
            # Apply stage-specific processing
            processor = self.stage_processors[self.current_stage]
            processed_image, visual_features = processor(image)
            
            # Update metrics based on processing
            self._update_metrics(visual_features)
            
            return {
                'success': True,
                'processed_image': processed_image,
                'features': visual_features,
                'metrics': self.metrics,
                'stage': self.current_stage.name
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def _process_newborn_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process vision for newborn stage (0-2 months)"""
        # Apply heavy blur and reduce contrast
        processed = self.transforms[DevelopmentStage.NEWBORN](image)
        
        # Extract basic features (mainly light/dark patterns and movement)
        with torch.no_grad():
            features = self.visual_attention(processed)
        
        return processed, {
            'light_sensitivity': features.mean().item(),
            'contrast_detection': features.std().item(),
            'feature_type': 'basic_patterns'
        }
        
    def _process_infant_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process vision for infant stage (2-6 months)"""
        # Apply moderate blur and begin color processing
        processed = self.transforms[DevelopmentStage.INFANT](image)
        
        # Focus on face-like patterns and color
        with torch.no_grad():
            features = self.visual_attention(processed)
            
        return processed, {
            'color_sensitivity': features.mean(dim=1).mean().item(),
            'face_detection': self.face_detection_focus,
            'feature_type': 'faces_and_colors'
        }
        
    def _process_early_toddler_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process vision for early toddler stage (6-12 months)"""
        processed = self.transforms[DevelopmentStage.EARLY_TODDLER](image)
        
        # Begin object recognition and depth perception
        with torch.no_grad():
            features = self.object_recognition(processed)
            
        return processed, {
            'object_recognition': features.max().item(),
            'depth_perception': self.metrics['depth_perception'],
            'feature_type': 'objects_and_depth'
        }
        
    def _process_late_toddler_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process vision for late toddler stage (12-24 months)"""
        processed = self.transforms[DevelopmentStage.LATE_TODDLER](image)
        
        # Enhanced object recognition and object permanence
        with torch.no_grad():
            features = self.object_recognition(processed)
            
        return processed, {
            'object_recognition': features.max().item(),
            'object_permanence': self.metrics['object_permanence'],
            'feature_type': 'detailed_objects'
        }
        
    def _process_preschool_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process vision for preschool stage (2-5 years)"""
        processed = self.transforms[DevelopmentStage.EARLY_PRESCHOOL](image)
        
        # Complex scene understanding and symbol recognition
        with torch.no_grad():
            features = self.object_recognition(processed)
            
        return processed, {
            'scene_understanding': features.mean().item(),
            'symbol_recognition': self.metrics['pattern_recognition'],
            'feature_type': 'scenes_and_symbols'
        }
        
    def _process_school_age_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process vision for school age stage (5-12 years)"""
        processed = self.transforms[DevelopmentStage.EARLY_CHILDHOOD](image)
        
        # Advanced pattern recognition and visual reasoning
        with torch.no_grad():
            features = self.object_recognition(processed)
            
        return processed, {
            'pattern_recognition': features.max().item(),
            'visual_reasoning': self.metrics['visual_memory'],
            'feature_type': 'patterns_and_reasoning'
        }
        
    def _process_adolescent_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process vision for adolescent stage (12+ years)"""
        processed = self.transforms[DevelopmentStage.PRE_ADOLESCENT](image)
        
        # Abstract visual concepts and complex pattern analysis
        with torch.no_grad():
            features = self.object_recognition(processed)
            
        return processed, {
            'abstract_recognition': features.mean().item(),
            'complex_patterns': features.max().item(),
            'feature_type': 'abstract_concepts'
        }
        
    def _process_adult_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process vision for adult stage (18+ years)"""
        processed = self.transforms[DevelopmentStage.YOUNG_ADULT](image)
        
        # Full visual processing capabilities
        with torch.no_grad():
            features = self.object_recognition(processed)
            
        return processed, {
            'visual_mastery': features.mean().item(),
            'complex_analysis': features.max().item(),
            'feature_type': 'full_visual_mastery'
        }
        
    def _update_metrics(self, visual_features: Dict):
        """Update vision development metrics based on processing results"""
        # Update metrics based on stage and features
        if self.current_stage == DevelopmentStage.NEWBORN:
            self.metrics['visual_acuity'] = min(0.2, self.metrics['visual_acuity'] + 0.01)
            
        elif self.current_stage == DevelopmentStage.INFANT:
            self.metrics['visual_acuity'] = min(0.4, self.metrics['visual_acuity'] + 0.02)
            self.metrics['color_perception'] = min(0.6, self.metrics['color_perception'] + 0.05)
            
        elif self.current_stage == DevelopmentStage.EARLY_TODDLER:
            self.metrics['depth_perception'] = min(0.7, self.metrics['depth_perception'] + 0.05)
            self.metrics['pattern_recognition'] = min(0.4, self.metrics['pattern_recognition'] + 0.03)
            
        elif self.current_stage == DevelopmentStage.LATE_TODDLER:
            self.metrics['object_permanence'] = min(0.8, self.metrics['object_permanence'] + 0.05)
            self.metrics['visual_memory'] = min(0.5, self.metrics['visual_memory'] + 0.03)
            
        elif self.current_stage == DevelopmentStage.EARLY_PRESCHOOL:
            self.metrics['pattern_recognition'] = min(0.7, self.metrics['pattern_recognition'] + 0.04)
            self.metrics['visual_memory'] = min(0.7, self.metrics['visual_memory'] + 0.04)
            
        elif self.current_stage == DevelopmentStage.LATE_PRESCHOOL:
            self.metrics['pattern_recognition'] = min(0.9, self.metrics['pattern_recognition'] + 0.03)
            self.metrics['visual_memory'] = min(0.8, self.metrics['visual_memory'] + 0.03)
            
        elif self.current_stage == DevelopmentStage.EARLY_CHILDHOOD:
            self.metrics['pattern_recognition'] = min(1.0, self.metrics['pattern_recognition'] + 0.02)
            self.metrics['visual_memory'] = min(0.9, self.metrics['visual_memory'] + 0.02)
            
        elif self.current_stage == DevelopmentStage.MIDDLE_CHILDHOOD:
            self.metrics['pattern_recognition'] = min(1.0, self.metrics['pattern_recognition'] + 0.02)
            self.metrics['visual_memory'] = min(0.9, self.metrics['visual_memory'] + 0.02)
            
        elif self.current_stage == DevelopmentStage.LATE_CHILDHOOD:
            self.metrics['pattern_recognition'] = min(1.0, self.metrics['pattern_recognition'] + 0.02)
            self.metrics['visual_memory'] = min(0.9, self.metrics['visual_memory'] + 0.02)
            
        elif self.current_stage == DevelopmentStage.PRE_ADOLESCENT:
            self.metrics['pattern_recognition'] = min(1.0, self.metrics['pattern_recognition'] + 0.02)
            self.metrics['visual_memory'] = min(0.9, self.metrics['visual_memory'] + 0.02)
            
        elif self.current_stage == DevelopmentStage.EARLY_TEEN:
            self.metrics['pattern_recognition'] = min(1.0, self.metrics['pattern_recognition'] + 0.02)
            self.metrics['visual_memory'] = min(0.9, self.metrics['visual_memory'] + 0.02)
            
        elif self.current_stage == DevelopmentStage.MID_TEEN:
            self.metrics['pattern_recognition'] = min(1.0, self.metrics['pattern_recognition'] + 0.02)
            self.metrics['visual_memory'] = min(0.9, self.metrics['visual_memory'] + 0.02)
            
        elif self.current_stage == DevelopmentStage.LATE_TEEN:
            self.metrics['pattern_recognition'] = min(1.0, self.metrics['pattern_recognition'] + 0.02)
            self.metrics['visual_memory'] = min(0.9, self.metrics['visual_memory'] + 0.02)
            
        elif self.current_stage == DevelopmentStage.YOUNG_ADULT:
            self.metrics['pattern_recognition'] = 1.0
            self.metrics['visual_memory'] = 1.0
            
        elif self.current_stage == DevelopmentStage.EARLY_TWENTIES:
            self.metrics['pattern_recognition'] = 1.0
            self.metrics['visual_memory'] = 1.0
            
        elif self.current_stage == DevelopmentStage.LATE_TWENTIES:
            self.metrics['pattern_recognition'] = 1.0
            self.metrics['visual_memory'] = 1.0
            
        # Update attention span across all stages
        self.metrics['attention_span'] = min(1.0, self.metrics['attention_span'] + 0.01)
        
    def get_development_summary(self) -> Dict[str, Any]:
        """Get summary of current vision development status"""
        return {
            'stage': self.current_stage.name,
            'stage_progress': self.stage_progress,
            'metrics': self.metrics,
            'capabilities': {
                'face_detection': self.face_detection_focus,
                'object_recognition': self.metrics['pattern_recognition'],
                'visual_processing': self.metrics['visual_acuity']
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current vision development status"""
        return {
            'stage': self.current_stage.name,
            'stage_progress': self.stage_progress,
            'metrics': self.metrics,
            'capabilities': {
                'object_recognition': self.object_recognition_accuracy,
                'face_recognition': self.face_recognition_accuracy,
                'depth_perception': self.depth_perception_accuracy,
                'color_recognition': self.color_recognition_accuracy,
                'motion_tracking': self.motion_tracking_accuracy
            }
        } 