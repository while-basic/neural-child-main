# conversation_system.py
# Created by Christopher Celaya
# Handles conversations between mother and baby LLMs

import json
from datetime import datetime
from llm_module import chat_completion
import torch
import re

class ConversationSystem:
    def __init__(self, logger):
        self.logger = logger
        self.conversation_history = []
        self.mother_personality = {
            'role': 'mother',
            'traits': ['nurturing', 'patient', 'encouraging', 'protective'],
            'goals': ['development', 'learning', 'emotional_bonding'],
            'speaking_style': {
                'perspective': 'first_person',
                'tone': 'warm_and_personal',
                'expressions': [
                    "I see that",
                    "I notice",
                    "I feel",
                    "Let me",
                    "I'm here",
                    "I'll help"
                ]
            }
        }
        self.baby_personality = {
            'role': 'baby',
            'traits': ['curious', 'developing', 'emotional', 'dependent'],
            'goals': ['learning', 'exploration', 'attachment']
        }
        
    def format_mother_prompt(self, stage, context):
        """Format the mother's prompt based on developmental stage and context"""
        return (
            f"You are a nurturing mother experiencing the following situation with your {stage.name.lower().replace('_', ' ')} child. "
            f"Current context: {context}\n\n"
            "Observe and react to:\n"
            "1. Changes in the environment (light, noise, temperature, chaos)\n"
            "2. Your child's emotional state and needs\n"
            "3. Your own emotional response to the situation\n\n"
            "Your responses should be:\n"
            "1. Unique and specific to the current environment and your child's state\n"
            "2. Always in first person, expressing YOUR observations and feelings\n"
            "3. Detailed about what you're seeing, feeling, and thinking\n"
            "4. Natural and varied - avoid repeating the same phrases\n"
            "5. Age-appropriate for your child's stage\n\n"
            "Response Guidelines:\n"
            "- Describe specific environmental details you notice\n"
            "- Share your emotional reactions to changes\n"
            "- Express how you plan to respond to your child's needs\n"
            "- Vary your language and expressions\n\n"
            "Structure your response as JSON with fields:\n"
            "   - content: your detailed, first-person observation and response\n"
            "   - emotion: your current emotional state\n"
            "   - intention: your specific parental intention\n"
            "   - developmental_focus: aspect of development you're targeting\n\n"
            "Example varied responses:\n"
            '{"content": "I feel the room getting warmer, little one. Your cheeks are flushed - let me move us to the cooler corner where the breeze is better.", "emotion": "attentive", "intention": "physical_comfort", "developmental_focus": "environmental_adaptation"}\n'
            '{"content": "I notice how the sudden noise startled you. Those are just the wind chimes dancing in the breeze - see how they make such pretty sounds?", "emotion": "reassuring", "intention": "learning_opportunity", "developmental_focus": "sensory_understanding"}\n'
            '{"content": "I see your eyes following the shadows on the wall. They\'re fascinating, aren\'t they? Let\'s watch how they move together.", "emotion": "engaged", "intention": "shared_discovery", "developmental_focus": "visual_tracking"}\n'
            '{"content": "I\'m noticing how peaceful the environment is right now. Your gentle breathing tells me you\'re feeling calm too.", "emotion": "serene", "intention": "emotional_attunement", "developmental_focus": "emotional_awareness"}\n\n'
            "Previous conversation: " + json.dumps(self.conversation_history[-3:] if self.conversation_history else [])
        )
        
    def format_baby_prompt(self, stage, context, emotional_state):
        """Format the baby's prompt based on developmental stage and emotional state"""
        return (
            f"You are a {stage.name.lower().replace('_', ' ')} child. "
            f"Your emotional state: {emotional_state}\n"
            f"Current context: {context}\n"
            "Your responses should be:\n"
            "1. Limited by your developmental stage\n"
            "2. Emotionally authentic\n"
            "3. Showing appropriate developmental behaviors\n"
            "4. Structured as a JSON with fields:\n"
            "   - content: your verbal or non-verbal response\n"
            "   - emotion: your emotional state\n"
            "   - need: your current need or desire\n"
            "   - learning: what you're trying to understand\n"
            "Previous conversation: " + json.dumps(self.conversation_history[-3:] if self.conversation_history else [])
        )
        
    def generate_mother_response(self, stage, baby_response, context):
        """Generate mother's response using the mother LLM"""
        # Create a more detailed context combining environment and baby's state
        detailed_context = (
            f"Environment: {context}\n"
            f"Baby's response: {baby_response}\n"
            f"Current stage: {stage.name}\n"
            f"Time: {datetime.now().strftime('%H:%M')}"
        )
        
        prompt = self.format_mother_prompt(stage, detailed_context)
        try:
            response = chat_completion(
                system_prompt=prompt,
                user_prompt=(
                    f"Observe and respond to the current situation:\n"
                    f"- Environment: {context}\n"
                    f"- Baby's state: {baby_response}\n"
                    f"Provide a complete JSON response with content, emotion, intention, and developmental_focus fields."
                ),
                structured_output=True,
                model="llama2"  # Explicitly using llama2 for mother
            )
            
            # If response is None or invalid, use fallback
            if response is None or not isinstance(response, dict):
                response = self._generate_fallback_mother_response()
            
            # Ensure response has required fields
            if not all(key in response for key in ['content', 'emotion', 'intention', 'developmental_focus']):
                response = self._generate_fallback_mother_response()
            
            # Get content and ensure it's a string
            content = str(response.get('content', ''))
            if not content:
                response = self._generate_fallback_mother_response()
                content = response['content']
            
            # Check for first-person perspective
            needs_first_person = True
            
            # Check if content already starts with first-person expressions
            for exp in self.mother_personality['speaking_style']['expressions']:
                if content.startswith(exp):
                    needs_first_person = False
                    break
            
            if needs_first_person:
                try:
                    # Get a new first-person version from LLM
                    first_person_prompt = (
                        "Convert this mother's response to first person, starting with 'I see', 'I notice', 'I feel', or similar:\n"
                        f"Original: {content}"
                    )
                    first_person_response = chat_completion(
                        system_prompt="You are a mother. Convert the response to first person perspective.",
                        user_prompt=first_person_prompt,
                        structured_output=True,
                        model="llama2"
                    )
                    
                    if isinstance(first_person_response, dict) and 'content' in first_person_response:
                        new_content = str(first_person_response.get('content', ''))
                        if new_content:
                            # Check if the new content starts with first-person expressions
                            for exp in self.mother_personality['speaking_style']['expressions']:
                                if new_content.startswith(exp):
                                    content = new_content
                                    needs_first_person = False
                                    break
                    
                    if needs_first_person:
                        content = f"I notice {content}"
                except Exception as e:
                    self.logger.log_error(e, "Error converting to first person")
                    content = f"I notice {content}"
            
            # Calculate emotional values based on LLM response
            emotion = str(response.get('emotion', '')).lower()
            formatted_response = {
                'content': content,
                'emotion': emotion,
                'intention': str(response.get('intention', 'comfort')),
                'developmental_focus': str(response.get('developmental_focus', 'emotional_security')),
                'joy': float(response.get('joy', 0.7 if 'joy' in emotion or 'happy' in emotion else 0.5)),
                'trust': float(response.get('trust', 0.8 if 'trust' in emotion or 'nurturing' in emotion else 0.6)),
                'fear': float(response.get('fear', 0.6 if 'fear' in emotion or 'worried' in emotion else 0.2)),
                'surprise': float(response.get('surprise', 0.6 if 'surprise' in emotion or 'curious' in emotion else 0.3))
            }
            
            # Record the interaction
            self.conversation_history.append({
                'speaker': 'mother',
                'response': formatted_response,
                'timestamp': datetime.now().isoformat(),
                'stage': stage.name
            })
            
            return formatted_response
                
        except Exception as e:
            self.logger.log_error(e, "Error generating mother response")
            return self._generate_fallback_mother_response()

    def _generate_fallback_mother_response(self):
        """Generate a safe fallback response"""
        return {
            'content': "I'm right here with you, my little one. I see you need me.",
            'emotion': 'nurturing',
            'intention': 'comfort',
            'developmental_focus': 'emotional_security',
            'joy': 0.5,
            'trust': 0.7,
            'fear': 0.2,
            'surprise': 0.2
        }
            
    def generate_baby_response(self, stage, mother_input, emotional_state, context):
        """Generate baby's response using the baby LLM with proper error handling"""
        try:
            # Extract mother's message from input
            if isinstance(mother_input, dict) and 'text' in mother_input:
                mother_message = mother_input['text']
            elif isinstance(mother_input, torch.Tensor):
                mother_message = "[PERCEIVED STIMULUS]"  # Handle tensor input
            else:
                mother_message = str(mother_input)
                
            prompt = self.format_baby_prompt(stage, context, emotional_state)
            
            try:
                response = chat_completion(
                    system_prompt=prompt,
                    user_prompt=f"Your mother just: {mother_message}",
                    structured_output=True,
                    model="llama2"  # Using llama2 for more structured responses
                )
            except Exception as e:
                self.logger.log_error(f"Error in chat completion: {str(e)}")
                response = None
            
            # If response is None or invalid, use fallback
            if response is None or not isinstance(response, dict):
                response = self._generate_fallback_baby_response(stage)
            
            # Ensure response has all required fields
            formatted_response = {
                'content': response.get('content', '[COOS]'),
                'emotion': response.get('emotion', 'neutral'),
                'need': response.get('need', 'comfort'),
                'learning': response.get('learning', 'basic_interaction'),
                'joy': 0.5 if response.get('emotion') in ['happy', 'excited'] else 0.3,
                'trust': 0.6 if response.get('emotion') in ['content', 'secure'] else 0.4,
                'fear': 0.3 if response.get('emotion') in ['anxious', 'distressed'] else 0.1,
                'surprise': 0.4 if response.get('emotion') in ['curious', 'interested'] else 0.2
            }
            
            self.conversation_history.append({
                'speaker': 'baby',
                'response': formatted_response,
                'timestamp': datetime.now().isoformat(),
                'stage': stage.name,
                'emotional_state': emotional_state
            })
            return formatted_response
            
        except Exception as e:
            self.logger.log_error(f"Error generating baby response: {str(e)}")
            return self._generate_fallback_baby_response(stage)
            
    def _generate_fallback_baby_response(self, stage):
        """Generate a safe fallback response for the baby based on stage"""
        if stage.value <= 2:  # Newborn or early infancy
            return {
                'content': '[COOS]',
                'emotion': 'neutral',
                'need': 'comfort',
                'learning': 'basic_interaction',
                'joy': 0.3,
                'trust': 0.4,
                'fear': 0.1,
                'surprise': 0.2
            }
        else:
            return {
                'content': 'Mama?',
                'emotion': 'seeking',
                'need': 'attention',
                'learning': 'social_interaction',
                'joy': 0.4,
                'trust': 0.6,
                'fear': 0.2,
                'surprise': 0.3
            }
            
    def analyze_interaction(self, mother_response, baby_response):
        """Analyze the interaction for developmental insights"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'interaction_quality': self._calculate_interaction_quality(mother_response, baby_response),
            'developmental_progress': self._assess_developmental_progress(baby_response),
            'emotional_alignment': self._measure_emotional_alignment(mother_response, baby_response),
            'learning_opportunities': self._identify_learning_opportunities(mother_response, baby_response)
        }
        return analysis
        
    def _calculate_interaction_quality(self, mother_response, baby_response):
        """Calculate the quality of the interaction"""
        # Implement scoring based on response appropriateness, emotional matching, etc.
        score = 0.0
        if all(k in mother_response for k in ['content', 'emotion', 'intention']):
            score += 0.5
        if all(k in baby_response for k in ['content', 'emotion', 'need']):
            score += 0.5
        return min(1.0, score)
        
    def _assess_developmental_progress(self, baby_response):
        """Assess developmental progress from baby's response"""
        progress_markers = {
            'communication': 0.0,
            'emotional': 0.0,
            'cognitive': 0.0,
            'social': 0.0
        }
        
        # Analyze response content for developmental markers
        if 'learning' in baby_response:
            progress_markers['cognitive'] = 0.6
        if 'emotion' in baby_response:
            progress_markers['emotional'] = 0.7
        if 'content' in baby_response and len(baby_response['content']) > 0:
            progress_markers['communication'] = 0.5
        if 'need' in baby_response and 'attention' in baby_response['need']:
            progress_markers['social'] = 0.8
            
        return progress_markers
        
    def _measure_emotional_alignment(self, mother_response, baby_response):
        """Measure emotional alignment between mother and baby"""
        mother_emotion = mother_response.get('emotion', 'neutral')
        baby_emotion = baby_response.get('emotion', 'neutral')
        
        # Simple emotional alignment check
        if mother_emotion == baby_emotion:
            return 1.0
        elif mother_emotion in ['nurturing', 'comforting'] and baby_emotion in ['distressed', 'seeking']:
            return 0.8
        else:
            return 0.5
            
    def _identify_learning_opportunities(self, mother_response, baby_response):
        """Identify learning opportunities in the interaction"""
        opportunities = []
        
        if 'developmental_focus' in mother_response:
            opportunities.append({
                'type': mother_response['developmental_focus'],
                'initiated_by': 'mother',
                'potential': 0.8
            })
            
        if 'learning' in baby_response:
            opportunities.append({
                'type': baby_response['learning'],
                'initiated_by': 'baby',
                'potential': 0.9
            })
            
        return opportunities
        
    def save_conversation(self, path):
        """Save the conversation history to a file"""
        conversation_data = {
            'history': self.conversation_history,
            'metadata': {
                'total_turns': len(self.conversation_history),
                'timestamp': datetime.now().isoformat(),
                'mother_personality': self.mother_personality,
                'baby_personality': self.baby_personality
            }
        }
        
        with open(path, 'w') as f:
            json.dump(conversation_data, f, indent=2) 