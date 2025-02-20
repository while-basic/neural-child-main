# digital_child.py
# Description: Digital child class for simulating child development
# Created by: Christopher Celaya

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum, auto

from vision_development import VisionDevelopment, DevelopmentStage
from development_logger import DevelopmentLogger

class DigitalChild:
    def __init__(self, name: str = "Neural Child", age_months: int = 0):
        """Initialize digital child with basic attributes"""
        self.name = name
        self.age_months = age_months
        self.development_stage = self._calculate_development_stage()
        
        # Initialize development systems
        self.vision = VisionDevelopment()
        self.logger = DevelopmentLogger()
        
        # Development metrics
        self.metrics = {
            'cognitive': 0.0,
            'motor': 0.0,
            'social': 0.0,
            'language': 0.0,
            'emotional': 0.0
        }
        
    def _calculate_development_stage(self) -> DevelopmentStage:
        """Calculate developmental stage based on age"""
        if self.age_months <= 3:
            return DevelopmentStage.NEWBORN
        elif self.age_months <= 6:
            return DevelopmentStage.INFANT
        elif self.age_months <= 12:
            return DevelopmentStage.EARLY_TODDLER
        elif self.age_months <= 18:
            return DevelopmentStage.LATE_TODDLER
        elif self.age_months <= 24:
            return DevelopmentStage.EARLY_PRESCHOOL
        elif self.age_months <= 36:
            return DevelopmentStage.LATE_PRESCHOOL
        elif self.age_months <= 48:
            return DevelopmentStage.EARLY_CHILDHOOD
        elif self.age_months <= 60:
            return DevelopmentStage.MIDDLE_CHILDHOOD
        elif self.age_months <= 72:
            return DevelopmentStage.LATE_CHILDHOOD
        elif self.age_months <= 144:
            return DevelopmentStage.PRE_ADOLESCENT
        elif self.age_months <= 168:
            return DevelopmentStage.EARLY_TEEN
        elif self.age_months <= 192:
            return DevelopmentStage.MID_TEEN
        elif self.age_months <= 216:
            return DevelopmentStage.LATE_TEEN
        elif self.age_months <= 252:
            return DevelopmentStage.YOUNG_ADULT
        elif self.age_months <= 300:
            return DevelopmentStage.EARLY_TWENTIES
        else:
            return DevelopmentStage.LATE_TWENTIES
            
    def process_visual_input(self, image_path: str) -> Dict[str, Any]:
        """Process visual input and update development metrics"""
        try:
            # Process image based on current development stage
            vision_result = self.vision.process_image(image_path, self.development_stage)
            
            # Log development progress
            self.logger.log_vision_development(self.development_stage, vision_result)
            
            return vision_result
            
        except Exception as e:
            print(f"Error processing visual input: {str(e)}")
            return {} 