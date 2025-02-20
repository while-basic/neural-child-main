# digital_child.py
# Description: Digital child class for neural child development
# Created by: Christopher Celaya

from enum import Enum
from typing import Dict, List, Optional, Any
import torch
from conversation_system import ConversationSystem
from development_stages import DevelopmentStage
from language_development import LanguageDevelopment
from vision_development import VisionDevelopment, VisionStage
from logger import DevelopmentLogger

class DigitalChild:
    def __init__(self, logger: DevelopmentLogger):
        """Initialize the digital child."""
        self.logger = logger
        self.conversation_system = ConversationSystem(logger)
        self.language_development = LanguageDevelopment(device='cpu')
        self.vision_development = VisionDevelopment(device='cpu')
        self.current_stage = DevelopmentStage.NEWBORN
        self.emotional_state = torch.zeros(4)  # [joy, trust, fear, surprise]
        self.update_self_concept()
        
    def update_self_concept(self):
        """Update self-concept based on current developmental stage."""
        stage_concepts = {
            DevelopmentStage.NEWBORN: {
                'age_range': '0-3 months',
                'identity': {
                    'self_reference': 'baby',
                    'awareness_level': 'minimal',
                    'social_role': 'infant'
                },
                'capabilities': {
                    'interests': ['sounds', 'faces', 'movement'],
                    'skills': ['looking', 'listening', 'crying'],
                    'understanding': 'basic sensory'
                }
            },
            DevelopmentStage.INFANT: {
                'age_range': '3-12 months',
                'identity': {
                    'self_reference': 'baby',
                    'awareness_level': 'basic',
                    'social_role': 'infant'
                },
                'capabilities': {
                    'interests': ['toys', 'people', 'exploration'],
                    'skills': ['crawling', 'babbling', 'grasping'],
                    'understanding': 'object permanence developing'
                }
            },
            DevelopmentStage.EARLY_TODDLER: {
                'age_range': '1-2 years',
                'identity': {
                    'self_reference': 'toddler',
                    'awareness_level': 'emerging',
                    'social_role': 'young child'
                },
                'capabilities': {
                    'interests': ['play', 'independence', 'exploration'],
                    'skills': ['walking', 'simple words', 'self-feeding'],
                    'understanding': 'basic concepts'
                }
            },
            DevelopmentStage.LATE_TODDLER: {
                'age_range': '2-3 years',
                'identity': {
                    'self_reference': 'toddler',
                    'awareness_level': 'developing',
                    'social_role': 'young child'
                },
                'capabilities': {
                    'interests': ['play', 'independence', 'exploration'],
                    'skills': ['running', 'sentences', 'self-feeding'],
                    'understanding': 'expanding concepts'
                }
            },
            DevelopmentStage.EARLY_PRESCHOOL: {
                'age_range': '3-4 years',
                'identity': {
                    'self_reference': 'kid',
                    'awareness_level': 'developing',
                    'social_role': 'preschooler'
                },
                'capabilities': {
                    'interests': ['imagination', 'friends', 'stories'],
                    'skills': ['conversation', 'drawing', 'playing games'],
                    'understanding': 'expanding knowledge'
                }
            },
            DevelopmentStage.LATE_PRESCHOOL: {
                'age_range': '4-5 years',
                'identity': {
                    'self_reference': 'kid',
                    'awareness_level': 'established',
                    'social_role': 'preschooler'
                },
                'capabilities': {
                    'interests': ['learning', 'friends', 'creativity'],
                    'skills': ['advanced conversation', 'art', 'complex games'],
                    'understanding': 'growing knowledge'
                }
            }
        }
        
        self_concept = stage_concepts.get(self.current_stage)
        if self_concept:
            self.conversation_system.update_self_concept(self_concept)
            self.logger.log_development({
                'event': 'self_concept_update',
                'stage': self.current_stage.name,
                'self_concept': self_concept
            })
    
    def set_stage(self, stage: DevelopmentStage, progress: float = 0.0):
        """Set the developmental stage and update all systems."""
        if stage != self.current_stage:
            self.current_stage = stage
            self.update_self_concept()
            self.update_vision_stage(stage)  # Update vision stage
            self.logger.log_development({
                'event': 'stage_change',
                'new_stage': stage.name,
                'vision_stage': self.vision_development.current_stage.name
            })
    
    def process_input(self, input_text: str, emotional_state: Optional[torch.Tensor] = None) -> Dict:
        """Process input based on current developmental stage and self-concept."""
        try:
            if emotional_state is None:
                emotional_state = torch.zeros(5)  # Default neutral emotional state
                
            # Process through language development
            language_response = self.language_development.process_input(
                input_text,
                self.current_stage
            )
            
            # Process through conversation system with self-concept awareness
            conversation_response = self.conversation_system.process_child_response(
                language_response['response'],
                emotional_state
            )
            
            return {
                'success': True,
                'response': conversation_response['response'],
                'stage': self.current_stage.name,
                'emotional_state': emotional_state.cpu().tolist(),
                'language_metrics': language_response.get('metrics', {}),
                'self_concept': self.conversation_system.self_concept
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'process_input',
                'input': input_text,
                'stage': self.current_stage.name
            })
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_visual_input(self, image: torch.Tensor) -> Dict[str, Any]:
        """Process visual input and update development."""
        try:
            # Process visual input through vision development system
            vision_result = self.vision_development.process_visual_input(image)
            
            # Update emotional state based on visual processing
            if vision_result['success']:
                features = vision_result['features']
                # Adjust emotional state based on visual features
                if 'face_detection' in features and features['face_detection'] > 0.7:
                    self.emotional_state[1] += 0.1  # Increase trust when seeing faces
                if features.get('feature_type') == 'basic_patterns':
                    self.emotional_state[3] += 0.05  # Increase surprise for new patterns
                    
                # Log vision development
                self.logger.log_development({
                    'event': 'vision_processing',
                    'stage': self.vision_development.current_stage.name,
                    'metrics': vision_result['metrics']
                })
                
            return vision_result
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'process_visual_input',
                'stage': self.current_stage.name
            })
            return {
                'success': False,
                'error': str(e)
            }
            
    def update_vision_stage(self, stage: DevelopmentStage):
        """Update vision stage based on developmental stage."""
        stage_to_vision = {
            DevelopmentStage.NEWBORN: VisionStage.NEWBORN,
            DevelopmentStage.INFANT: VisionStage.INFANT,
            DevelopmentStage.EARLY_TODDLER: VisionStage.EARLY_TODDLER,
            DevelopmentStage.LATE_TODDLER: VisionStage.LATE_TODDLER,
            DevelopmentStage.EARLY_PRESCHOOL: VisionStage.PRESCHOOL,
            DevelopmentStage.LATE_PRESCHOOL: VisionStage.PRESCHOOL,
            DevelopmentStage.EARLY_CHILDHOOD: VisionStage.SCHOOL_AGE,
            DevelopmentStage.MIDDLE_CHILDHOOD: VisionStage.SCHOOL_AGE,
            DevelopmentStage.LATE_CHILDHOOD: VisionStage.SCHOOL_AGE,
            DevelopmentStage.PRE_ADOLESCENT: VisionStage.ADOLESCENT,
            DevelopmentStage.EARLY_TEEN: VisionStage.ADOLESCENT,
            DevelopmentStage.MID_TEEN: VisionStage.ADOLESCENT,
            DevelopmentStage.LATE_TEEN: VisionStage.ADOLESCENT,
            DevelopmentStage.YOUNG_ADULT: VisionStage.ADULT,
            DevelopmentStage.EARLY_TWENTIES: VisionStage.ADULT,
            DevelopmentStage.LATE_TWENTIES: VisionStage.ADULT
        }
        
        vision_stage = stage_to_vision.get(stage, VisionStage.ADULT)
        self.vision_development.current_stage = vision_stage
        
    def get_development_status(self) -> Dict[str, Any]:
        """Get current development status including vision development."""
        try:
            language_status = self.language_development.get_status()
            conversation_analysis = self.conversation_system.analyze_conversation()
            vision_summary = self.vision_development.get_development_summary()
            
            return {
                'stage': self.current_stage.name,
                'self_concept': self.conversation_system.self_concept,
                'language_development': language_status,
                'vision_development': vision_summary,
                'conversation_metrics': conversation_analysis
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'get_development_status'
            })
            return {
                'error': str(e)
            }
    
    def save_state(self, filepath: str):
        """Save current state including self-concept."""
        try:
            self.conversation_system.save_conversation(filepath)
            self.logger.log_development({
                'event': 'state_saved',
                'filepath': filepath
            })
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'save_state',
                'filepath': filepath
            })
    
    def load_state(self, filepath: str):
        """Load saved state including self-concept."""
        try:
            self.conversation_system.load_conversation(filepath)
            self.logger.log_development({
                'event': 'state_loaded',
                'filepath': filepath
            })
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'load_state',
                'filepath': filepath
            }) 