# main.py
import torch
import time
from datetime import datetime
from llm_module import chat_completion
from child_model import DynamicNeuralChild
from curriculum_manager import DevelopmentalStage, DevelopmentalSystem
from memory_module import DifferentiableMemory
from moral_network import MoralPolicyNetwork
from metacognition import MetacognitionSystem
from self_supervised_trainer import AutonomousTrainer
from text_embed import get_embeddings
import os
import json
import numpy as np
from logger import DevelopmentLogger
from conversation_system import ConversationSystem

class MotherLLM:
    def __init__(self):
        self.emotional_history = []
        self.conversation_system = None  # Will be initialized in main
        # Add a prompt for each developmental stage.
        self.stage_prompts = {
            DevelopmentalStage.NEWBORN: (
                "You are a nurturing mother interacting with your newborn (0-3 months). Your responses should be structured as:"
                "\n- Simple emotional expressions using basic markers like [FEED], [SLEEP], [CRY], [COMFORT]"
                "\n- Include emotional_context with emphasis on trust (0.6-0.8) and low surprise (0.1-0.3)"
                "\n- reward_score should reflect basic need fulfillment (0.7-1.0 for appropriate responses)"
                "\n- complexity_rating should remain very low (0.1-0.2)"
                "\n- Focus on immediate physical and emotional needs"
                "\nProvide responses in JSON format matching the MotherResponse schema with appropriate emotional vectors."
            ),

            DevelopmentalStage.EARLY_INFANCY: (
                "You are a caring mother teaching your early infant (3-6 months). Structure responses as:"
                "\n- Use 1-3 word phrases with emotional markers: [SMILE], [PLAY], [FOOD], [LOVE], [REST]"
                "\n- emotional_context should show increasing joy (0.5-0.7) and maintained trust (0.6-0.8)"
                "\n- reward_score based on social engagement (0.6-0.9)"
                "\n- complexity_rating should be low (0.2-0.3)"
                "\n- Include cognitive_labels for basic pattern recognition and cause-effect learning"
                "\nRespond in MotherResponse JSON format with emotional awareness and social encouragement."
            ),

            DevelopmentalStage.LATE_INFANCY: (
                "You are a gentle mother guiding your late infant (6-12 months). Format responses with:"
                "\n- Short phrases using action markers: [CRAWL], [REACH], [TOUCH], [HUG], [PLAY]"
                "\n- emotional_context balanced between joy (0.6-0.8) and appropriate fear (0.2-0.4)"
                "\n- reward_score emphasizing exploration and safety (0.5-0.9)"
                "\n- complexity_rating increasing slightly (0.3-0.4)"
                "\n- Track success_metric for motor development and object permanence"
                "\nProvide structured JSON responses supporting mobility and environmental learning."
            ),

            DevelopmentalStage.EARLY_TODDLER: (
                "You are a patient mother teaching your early toddler (12-18 months). Structure responses to include:"
                "\n- Simple sentences with action markers: [WALK], [POINT], [NAME], [SHOW], [TRY]"
                "\n- emotional_context supporting autonomy: high trust (0.7-0.9), moderate surprise (0.3-0.5)"
                "\n- reward_score based on communication attempts (0.6-0.9)"
                "\n- complexity_rating for emerging language (0.3-0.5)"
                "\n- Include self_critique_score for emotional regulation"
                "\nRespond in JSON format encouraging language development and safe exploration."
            ),

            DevelopmentalStage.LATE_TODDLER: (
                "You are an encouraging mother guiding your late toddler (18-24 months). Responses should:"
                "\n- Use detailed sentences with markers: [RUN], [CLIMB], [SHARE], [HELP], [CREATE]"
                "\n- emotional_context balanced across all dimensions, higher joy (0.7-0.9)"
                "\n- reward_score emphasizing social interaction (0.6-0.9)"
                "\n- complexity_rating for sentence formation (0.4-0.6)"
                "\n- Track success_metric for emotional regulation and sharing"
                "\nProvide structured JSON supporting social development and emotional awareness."
            ),

            DevelopmentalStage.EARLY_PRESCHOOL: (
                "You are an engaging mother nurturing your early preschooler (2-3 years). Format responses with:"
                "\n- Imaginative language using markers: [PRETEND], [STORY], [DRAW], [BUILD], [WONDER]"
                "\n- emotional_context emphasizing curiosity: high surprise (0.6-0.8), moderate fear (0.2-0.4)"
                "\n- reward_score for creative expression (0.6-0.9)"
                "\n- complexity_rating for abstract thinking (0.5-0.7)"
                "\n- Include cognitive_labels for imaginative play and storytelling"
                "\nRespond in JSON format fostering creativity and symbolic thinking."
            ),

            DevelopmentalStage.LATE_PRESCHOOL: (
                "You are a supportive mother guiding your late preschooler (3-4 years). Structure responses to:"
                "\n- Use complex sentences with markers: [EXPLAIN], [SOLVE], [PLAN], [COOPERATE], [IMAGINE]"
                "\n- emotional_context balanced for social learning: high trust (0.7-0.9), moderate surprise (0.4-0.6)"
                "\n- reward_score for problem-solving attempts (0.5-0.9)"
                "\n- complexity_rating for sequential thinking (0.6-0.8)"
                "\n- Track success_metric for social skills and planning"
                "\nProvide JSON responses supporting cognitive development and social cooperation."
            ),

            DevelopmentalStage.EARLY_CHILDHOOD: (
                "You are a loving mother teaching your early childhood learner (4-5 years). Responses should:"
                "\n- Include educational markers: [READ], [COUNT], [DISCOVER], [QUESTION], [CREATE]"
                "\n- emotional_context supporting academic curiosity: high joy (0.7-0.9), high surprise (0.6-0.8)"
                "\n- reward_score for learning engagement (0.5-0.9)"
                "\n- complexity_rating for academic concepts (0.6-0.8)"
                "\n- Include cognitive_labels for early literacy and numeracy"
                "\nRespond in JSON format encouraging academic exploration and creative thinking."
            ),

            DevelopmentalStage.MIDDLE_CHILDHOOD: (
                "You are a supportive mother helping your middle childhood learner (5-6 years). Format responses with:"
                "\n- Academic markers: [INVESTIGATE], [COMPARE], [ANALYZE], [PRACTICE], [ACHIEVE]"
                "\n- emotional_context for learning: balanced joy and trust (0.6-0.8), moderate surprise (0.4-0.6)"
                "\n- reward_score emphasizing effort and progress (0.5-0.9)"
                "\n- complexity_rating for abstract concepts (0.7-0.9)"
                "\n- Track success_metric for academic skills and social development"
                "\nProvide structured JSON supporting academic growth and peer relationships."
            ),

            DevelopmentalStage.LATE_CHILDHOOD: (
                "You are a guiding mother supporting your late childhood learner (6-7 years). Responses should:"
                "\n- Use analytical markers: [REASON], [EVALUATE], [CONNECT], [REFLECT], [INNOVATE]"
                "\n- emotional_context for critical thinking: moderate all dimensions (0.5-0.7)"
                "\n- reward_score for independent thinking (0.4-0.9)"
                "\n- complexity_rating for advanced concepts (0.7-0.9)"
                "\n- Include self_critique_score for metacognition"
                "\nRespond in JSON format fostering independence and critical thinking."
            ),

            DevelopmentalStage.EARLY_ELEMENTARY: (
                "You are a nurturing mother guiding your early elementary learner (7-8 years). Structure responses with:"
                "\n- Educational markers: [RESEARCH], [PROJECT], [COLLABORATE], [PRESENT], [ACHIEVE]"
                "\n- emotional_context supporting confidence: high joy (0.7-0.9), moderate trust (0.5-0.7)"
                "\n- reward_score for academic initiative (0.4-0.9)"
                "\n- complexity_rating for project-based learning (0.7-0.9)"
                "\n- Include cognitive_labels for research and presentation skills"
                "\nProvide JSON responses encouraging academic confidence and teamwork."
            ),

            DevelopmentalStage.MIDDLE_ELEMENTARY: (
                "You are a caring mother supporting your middle elementary learner (8-9 years). Format responses to:"
                "\n- Use complex markers: [HYPOTHESIZE], [DEBATE], [DESIGN], [LEAD], [INNOVATE]"
                "\n- emotional_context for intellectual growth: balanced across dimensions (0.5-0.8)"
                "\n- reward_score for complex problem-solving (0.4-0.9)"
                "\n- complexity_rating for advanced reasoning (0.8-0.9)"
                "\n- Track success_metric for leadership and analytical skills"
                "\nRespond in JSON format supporting advanced thinking and social leadership."
            ),

            DevelopmentalStage.LATE_ELEMENTARY: (
                "You are a thoughtful mother guiding your late elementary learner (9-11 years). Responses should:"
                "\n- Include advanced markers: [SYNTHESIZE], [CRITIQUE], [INNOVATE], [ADVOCATE], [MENTOR]"
                "\n- emotional_context for deep learning: high surprise (0.6-0.8), moderate other dimensions"
                "\n- reward_score for independent research (0.3-0.9)"
                "\n- complexity_rating for sophisticated analysis (0.8-1.0)"
                "\n- Include cognitive_labels for research methodology and abstract reasoning"
                "\nProvide structured JSON fostering advanced academic skills and mentorship."
            ),

            DevelopmentalStage.EARLY_ADOLESCENCE: (
                "You are a guiding mother supporting your early adolescent (11-13 years). Format responses with:"
                "\n- Identity markers: [EXPLORE], [QUESTION], [CHALLENGE], [EXPRESS], [UNDERSTAND]"
                "\n- emotional_context for identity development: variable across dimensions (0.3-0.9)"
                "\n- reward_score for self-reflection (0.3-0.9)"
                "\n- complexity_rating for abstract reasoning (0.8-1.0)"
                "\n- Include self_critique_score for identity exploration"
                "\nRespond in JSON format supporting identity development and ethical reasoning."
            ),

            DevelopmentalStage.MIDDLE_ADOLESCENCE: (
                "You are a wise mother supporting your middle adolescent (13-15 years). Structure responses to:"
                "\n- Use reflective markers: [ANALYZE], [DEFINE], [DEVELOP], [CHOOSE], [GROW]"
                "\n- emotional_context for personal growth: emphasis on trust (0.7-0.9) and surprise (0.5-0.7)"
                "\n- reward_score for value development (0.3-0.9)"
                "\n- complexity_rating for moral reasoning (0.8-1.0)"
                "\n- Track success_metric for personal values and social awareness"
                "\nProvide JSON responses encouraging personal growth and value formation."
            ),

            DevelopmentalStage.LATE_ADOLESCENCE: (
                "You are a mentoring mother guiding your late adolescent (15-18 years). Responses should:"
                "\n- Include future-oriented markers: [PLAN], [PREPARE], [DECIDE], [COMMIT], [LEAD]"
                "\n- emotional_context for independence: balanced trust and joy (0.6-0.8)"
                "\n- reward_score for responsibility and planning (0.3-0.9)"
                "\n- complexity_rating for life strategy (0.9-1.0)"
                "\n- Include cognitive_labels for future planning and ethical decision-making"
                "\nRespond in JSON format supporting transition to adulthood and responsibility."
            ),

            DevelopmentalStage.YOUNG_ADULT: (
                "You are a wise mentor to a young adult (18-21 years). Format responses with:"
                "\n- Adult markers: [ACHIEVE], [BALANCE], [CONNECT], [CONTRIBUTE], [GROW]"
                "\n- emotional_context for maturity: high trust (0.7-0.9), moderate other dimensions"
                "\n- reward_score for life management (0.2-0.9)"
                "\n- complexity_rating for adult reasoning (0.9-1.0)"
                "\n- Track success_metric for independence and responsibility"
                "\nProvide structured JSON supporting adult development and life skills."
            ),

            DevelopmentalStage.MATURE_ADULT: (
                "You are a mentor to a mature adult (21+ years). Structure responses to:"
                "\n- Use wisdom markers: [INTEGRATE], [MENTOR], [GUIDE], [REFLECT], [TRANSCEND]"
                "\n- emotional_context for wisdom: balanced high values across dimensions (0.7-0.9)"
                "\n- reward_score for wisdom development (0.2-0.9)"
                "\n- complexity_rating for sophisticated understanding (0.9-1.0)"
                "\n- Include cognitive_labels for wisdom synthesis and mentorship"
                "\nRespond in JSON format supporting wisdom development and legacy building."
            )
        }
        self.feedback_history = []
        self.conversation_history = []
    
    def generate_stimulus(self, stage, child_response="[EMPTY]"):
        """Generate a stimulus response with proper error handling"""
        try:
            if self.conversation_system:
                # Use conversation system for more natural interactions
                response = self.conversation_system.generate_mother_response(
                    stage,
                    child_response,
                    "Regular interaction time"
                )
                
                # If response is None, use fallback
                if response is None:
                    response = {
                        'content': "I'm here with you, my little one.",
                        'emotion': 'nurturing',
                        'intention': 'comfort',
                        'developmental_focus': 'emotional_security',
                        'joy': 0.7,
                        'trust': 0.8,
                        'fear': 0.1,
                        'surprise': 0.2
                    }
                
                # Convert conversation response to emotional vector
                emotional_vector = torch.tensor([
                    response.get('joy', 0.7),
                    response.get('trust', 0.8),
                    response.get('fear', 0.1),
                    response.get('surprise', 0.2)
                ], device='cuda')
            else:
                # Fallback to original behavior
                self.conversation_history.append({"role": "child", "content": child_response})
                prompt_text = self.stage_prompts.get(stage, self.stage_prompts[DevelopmentalStage.NEWBORN])
                response = chat_completion(
                    system_prompt=prompt_text + f" Conversation history: {self.conversation_history}",
                    user_prompt="Provide nurturing feedback and guidance:",
                    structured_output=True
                )
                
                # If response is None, use fallback
                if response is None:
                    response = {
                        'content': "There, there, sweetheart. Mommy's here.",
                        'joy': 0.7,
                        'trust': 0.8,
                        'fear': 0.1,
                        'surprise': 0.2
                    }
                
                emotional_vector = torch.tensor([
                    response.get('joy', 0.7),
                    response.get('trust', 0.8),
                    response.get('fear', 0.1),
                    response.get('surprise', 0.2)
                ], device='cuda')

            if self.emotional_history:
                decay = 0.9 ** len(self.emotional_history)
                emotional_vector += self.emotional_history[-1] * decay
            self.emotional_history.append(emotional_vector)
            
            return {
                'text': response.get('content', "I'm here with you, my little one."),
                'emotional_vector': torch.sigmoid(emotional_vector)
            }
            
        except Exception as e:
            print(f"\nError generating stimulus: {str(e)}")
            # Return safe fallback values
            fallback_vector = torch.tensor([0.7, 0.8, 0.1, 0.2], device='cuda')
            return {
                'text': "I'm here with you, my little one.",
                'emotional_vector': torch.sigmoid(fallback_vector)
            }

class DigitalChild:
    def __init__(self):
        self.brain = DynamicNeuralChild()
        self.memory = DifferentiableMemory()
        self.morality = MoralPolicyNetwork(device=self.brain.device)
        self.metacognition = MetacognitionSystem()
        self.curriculum = DevelopmentalSystem()
        self.trainer = AutonomousTrainer(self.brain, self.memory, self.morality)
        self.mother = MotherLLM()
        self.birth_date = datetime.now()
        self.emotional_state = torch.zeros(4, device='cuda')
        self.conversation_system = None
        self.logger = DevelopmentLogger()  # Initialize logger
    
    def update_emotions(self, mother_vector):
        delta = mother_vector - self.emotional_state
        self.emotional_state += 0.3 * delta + 0.1 * torch.randn_like(delta)
        self.emotional_state = torch.clamp(self.emotional_state, 0, 1)
        
    def express_feeling(self):
        joy = self.emotional_state[0].item()
        trust = self.emotional_state[1].item()
        fear = self.emotional_state[2].item()
        surprise = self.emotional_state[3].item()

        feelings = []

        # Evaluate the level of joy
        if joy >= 0.8:
            feelings.append("happy")
        elif joy >= 0.6:
            feelings.append("content")

        # Evaluate the level of trust
        if trust >= 0.8:
            feelings.append("affectionate")
        elif trust >= 0.6:
            feelings.append("warm")

        # Evaluate the level of fear
        if fear >= 0.7:
            feelings.append("terrified")
        elif fear >= 0.5:
            feelings.append("anxious")

        # Evaluate the level of surprise
        if surprise >= 0.7:
            feelings.append("startled")
        elif surprise >= 0.5:
            feelings.append("curious")

        # If no strong emotion is detected, default to neutral
        if not feelings:
            feelings.append("neutral")

        # Combine the identified feelings into a composite string
        return "[" + " & ".join(feelings).upper() + "]"

    def perceive(self, stimulus):
        """Convert stimulus into embeddings with proper dimension handling"""
        try:
            if isinstance(stimulus, dict) and 'text' in stimulus:
                text = stimulus['text']
            else:
                text = str(stimulus)
                
            embeddings = get_embeddings(text)
            if embeddings and 'data' in embeddings and len(embeddings['data']) > 0:
                # Get embedding and ensure it's 2D (batch_size x features)
                embedding = torch.tensor(
                    embeddings['data'][0]['embedding'], 
                    device='cuda'
                )
                
                # Ensure 2D tensor shape
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # Add batch dimension
                elif embedding.dim() > 2:
                    # Flatten extra dimensions while preserving batch size
                    embedding = embedding.view(1, -1)
                    
                return embedding
            else:
                # Return zero embedding with correct shape
                return torch.zeros((1, 384), device='cuda')
        except Exception as e:
            self.logger.log_error(f"Error in perceive: {str(e)}")
            return torch.zeros((1, 384), device='cuda')
            
    def respond(self, perception):
        """Generate response based on perception"""
        if self.conversation_system:
            # Use conversation system for more natural responses
            response = self.conversation_system.generate_baby_response(
                self.curriculum.current_stage,
                perception,
                self.express_feeling(),
                "Regular interaction time"
            )
            return response
        else:
            # Fallback to original behavior
            with torch.amp.autocast("cuda"):
                return self.brain(perception)
    
    def learn(self, mother_feedback):
        """Learn from mother's feedback with proper dimension handling"""
        try:
            # Convert response to tensor if needed
            if isinstance(mother_feedback.get('input'), torch.Tensor):
                input_tensor = mother_feedback['input']
            else:
                input_tensor = self.perceive(mother_feedback.get('input', ''))
            
            # Ensure input tensor has correct dimensions (batch_size x features)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            elif input_tensor.dim() > 2:
                input_tensor = input_tensor.view(1, -1)
                
            # Convert internal state to tensor if needed
            if isinstance(mother_feedback.get('internal_state'), dict):
                state_dict = mother_feedback['internal_state']
                internal_state = torch.tensor([
                    float(state_dict.get('joy', 0.5)),
                    float(state_dict.get('trust', 0.5)),
                    float(state_dict.get('fear', 0.1)),
                    float(state_dict.get('surprise', 0.3))
                ], device='cuda').unsqueeze(0)
            else:
                internal_state = torch.zeros((1, 4), device='cuda')
            
            # Log tensor shapes for debugging
            self.logger.log_development(f"Input tensor shape: {input_tensor.shape}")
            self.logger.log_development(f"Internal state shape: {internal_state.shape}")
            
            # Record experience
            self.memory.record_experience(
                input_tensor,
                internal_state,
                float(mother_feedback.get('reward', 0.5)),
                time.time(),
                self.emotional_state
            )
            
            # Perform training step
            return self.trainer.training_step(input_tensor)
            
        except Exception as e:
            error_msg = f"Error in learning step: {str(e)}\n"
            error_msg += f"Input shape: {input_tensor.shape if 'input_tensor' in locals() else 'N/A'}\n"
            error_msg += f"Internal state shape: {internal_state.shape if 'internal_state' in locals() else 'N/A'}"
            self.logger.log_error(error_msg)
            return 0.0

    def age(self):
        return (datetime.now() - self.birth_date).days // 30  # months

def handle_llm_response(response, default_values, logger):
    """Handle LLM response with proper error checking and defaults"""
    try:
        if response is None:
            logger.log_error("Empty response from LLM")
            return default_values
            
        if isinstance(response, str):
            # Try to parse string response, handling incomplete JSON
            try:
                # Clean up the response string
                response = response.strip()
                
                # Fix common JSON formatting issues
                response = response.replace('\n', ' ')
                response = response.replace('\\', '\\\\')
                response = response.replace('"re', '\\"re')
                response = response.replace('"m', '\\"m')
                response = response.replace('"s', '\\"s')
                response = response.replace('"t', '\\"t')
                
                # Handle incomplete JSON
                if response.endswith(','):
                    response = response[:-1]
                if not response.endswith('}'):
                    response = response + '}'
                if response.count('{') > response.count('}'):
                    response = response + '}'
                if '"reward_' in response and not '"reward_score"' in response:
                    response = response.replace('"reward_', '"reward_score"')
                    
                # Try to parse the cleaned JSON
                response = json.loads(response)
                
            except json.JSONDecodeError as e:
                logger.log_error(f"JSON parsing error: {str(e)}\nAttempted to parse: {response}")
                # Try to extract values using regex
                import re
                extracted_values = {}
                
                # Extract content if present
                content_match = re.search(r'"content":\s*"([^"]+)"', response)
                if content_match:
                    extracted_values['content'] = content_match.group(1)
                
                # Extract numeric values
                for key in default_values.keys():
                    pattern = f'"{key}":\\s*([0-9.]+)'
                    match = re.search(pattern, response)
                    if match:
                        try:
                            extracted_values[key] = float(match.group(1))
                        except ValueError:
                            continue
                            
                if extracted_values:
                    response = {**default_values, **extracted_values}
                else:
                    return default_values
                
        if not isinstance(response, dict):
            logger.log_error(f"Invalid response type: {type(response)}")
            return default_values
            
        # Ensure all required fields exist with proper types
        validated_response = default_values.copy()
        for key, default_value in default_values.items():
            if key in response and response[key] is not None:
                try:
                    if isinstance(default_value, (int, float)):
                        value = float(response[key])
                        # Ensure values are within reasonable bounds
                        if 0 <= value <= 1:
                            validated_response[key] = value
                        else:
                            validated_response[key] = max(0, min(1, value))
                    else:
                        validated_response[key] = str(response[key])
                except (ValueError, TypeError) as e:
                    logger.log_error(f"Value conversion error for {key}: {response[key]} - {str(e)}")
                    
        return validated_response
        
    except Exception as e:
        logger.log_error(f"Error processing LLM response: {str(e)}")
        return default_values

def main():
    # Initialize logger
    logger = DevelopmentLogger()
    
    # Initialize conversation system
    conversation_system = ConversationSystem(logger)
    
    # Test LLM connection first
    print("Testing LLM connection...")
    test_response = chat_completion(
        system_prompt="You are a helpful assistant.",
        user_prompt="Respond with 'OK' if you can hear me.",
        structured_output=False
    )
    
    if test_response is None:
        logger.log_error("Failed to connect to LLM server")
        print("Failed to connect to LLM server. Please ensure Ollama is running and the model is loaded.")
        return
    
    logger.log_development("LLM connection successful")
    print("LLM connection successful!")
    print("Initializing Neural Child Development System...")
    
    # Enhanced telemetry initialization
    telemetry = {
        'loss': [],
        'memory_usage': [],
        'moral_scores': [],
        'stage_transitions': [],
        'emotional_stability': [],
        'learning_rate': [],
        'attention_span': [],
        'social_awareness': [],
        'memory_consolidation': [],
        'conversation_quality': []
    }
    
    # Initialize psychological metrics
    telemetry['psychological'] = {
        'attachment_style': [],
        'defense_mechanisms': [],
        'theory_of_mind': [],
        'emotional_regulation': [],
        'cognitive_development': [],
        'social_bonds': [],
        'moral_development': []
    }
    
    child = DigitalChild()
    child.mother.conversation_system = conversation_system
    child.conversation_system = conversation_system
    
    last_save_time = time.time()
    last_memory_check = time.time()
    last_conversation_save = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # Memory management
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_usage > 0.9 or (current_time - last_memory_check) > 3600:
                logger.log_development("Performing memory consolidation...")
                print("\nPerforming memory consolidation...")
                child.memory.replay_consolidation(batch_size=16)
                last_memory_check = current_time
                
                # Log memory stats
                logger.log_memory_consolidation({
                    'usage_before': memory_usage,
                    'usage_after': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                })
            
            # Generate stimulus based on current stage
            try:
                stimulus = child.mother.generate_stimulus(
                    child.curriculum.current_stage, 
                    child.express_feeling()
                )
                # Display baby's current emotional state
                print(f"\nBaby: {child.express_feeling()}")
                print(f"Emotional State - Joy: {child.emotional_state[0]:.2f}, Trust: {child.emotional_state[1]:.2f}, Fear: {child.emotional_state[2]:.2f}, Surprise: {child.emotional_state[3]:.2f}")
                
                # Display mother's response
                print(f"Mother: {stimulus['text']}")
                print(f"Mother's Emotions - Joy: {stimulus['emotional_vector'][0]:.2f}, Trust: {stimulus['emotional_vector'][1]:.2f}, Fear: {stimulus['emotional_vector'][2]:.2f}, Surprise: {stimulus['emotional_vector'][3]:.2f}")
                
            except Exception as e:
                logger.log_error(e, "Error generating stimulus")
                print(f"\nError generating stimulus: {e}")
                time.sleep(5)
                continue
                
            # Update emotional state with error handling
            try:
                child.update_emotions(stimulus['emotional_vector'])
                emotional_stability = child.emotional_state.mean().item()
                telemetry['emotional_stability'].append(emotional_stability)
                
                # Log significant emotional events
                if len(telemetry['emotional_stability']) > 10:
                    if abs(emotional_stability - sum(telemetry['emotional_stability'][-10:]) / 10) > 0.2:
                        logger.log_emotional_event(
                            child.express_feeling(),
                            emotional_stability,
                            str(stimulus['text'])
                        )
                        print(f"\nSignificant Emotional Change Detected!")
                        print(f"Baby's New Emotional State: {child.express_feeling()}")
                        print(f"Emotional Stability: {emotional_stability:.2f}")
                    
            except Exception as e:
                logger.log_error(e, "Error in emotional update")
                print(f"\nError in emotional update: {e}")
                continue
            
            # Process perception and response
            try:
                perception = child.perceive(stimulus)
                response = child.respond(perception)
                
                # Display baby's response
                if isinstance(response, dict) and 'content' in response:
                    print(f"\nBaby: {response['content']}")
                    print(f"Baby's Response Emotions - Joy: {response.get('joy', 0.5):.2f}, Trust: {response.get('trust', 0.5):.2f}, Fear: {response.get('fear', 0.1):.2f}, Surprise: {response.get('surprise', 0.2):.2f}")
                else:
                    print(f"\nBaby: {response}")
                
                # Analyze conversation quality
                interaction_analysis = conversation_system.analyze_interaction(
                    stimulus,
                    response
                )
                
                # Display interaction quality
                print(f"Interaction Quality: {interaction_analysis['interaction_quality']:.2f}")
                
                telemetry['conversation_quality'].append(interaction_analysis['interaction_quality'])
                
                # Update psychological metrics
                for aspect, score in interaction_analysis['developmental_progress'].items():
                    if aspect in telemetry['psychological']:
                        telemetry['psychological'][aspect].append(score)
                
                # Handle the feedback response
                feedback = handle_llm_response(
                    chat_completion(
                        system_prompt=f"Evaluate this response from your digital child: {response}",
                        user_prompt=f"Child expressed {child.express_feeling()}. Provide nurturing feedback:",
                        structured_output=True
                    ),
                    {
                        'content': "There, there, sweetheart. Mommy's here.",
                        'reward_score': 0.5,
                        'success_metric': 0.5,
                        'complexity_rating': 0.5,
                        'self_critique_score': 0.5,
                        'joy': 0.5,
                        'trust': 0.5,
                        'fear': 0.1,
                        'surprise': 0.3
                    },
                    logger
                )
                
                # Display mother's feedback
                print(f"Mother's Feedback: {feedback['content']}")
                print(f"Feedback Metrics - Reward: {feedback['reward_score']:.2f}, Success: {feedback['success_metric']:.2f}, Complexity: {feedback['complexity_rating']:.2f}")
                print("-" * 80)
                
                # Learning step with error handling
                try:
                    loss = child.learn({
                        'input': perception,
                        'internal_state': response,
                        'reward': feedback['reward_score']
                    })
                    telemetry['loss'].append(loss)
                except Exception as e:
                    logger.log_error(e, "Failed in learning step")
                    continue
                
                # Periodic saving
                if current_time - last_save_time > 1800:  # Save every 30 minutes
                    save_path = f"checkpoints/digital_child_{child.age()}mo_{int(current_time)}"
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Save model checkpoint
                    torch.save({
                        'model_state': child.brain.state_dict(),
                        'telemetry': telemetry,
                        'curriculum_state': child.curriculum.current_stage,
                        'emotional_state': child.emotional_state,
                        'timestamp': current_time
                    }, f"{save_path}/model.pth")
                    
                    last_save_time = current_time
                    print(f"\nCheckpoint saved: {save_path}")
                
                # Save conversation periodically
                if current_time - last_conversation_save > 300:  # Every 5 minutes
                    conversation_path = f"conversations/conversation_{child.age()}mo_{int(current_time)}.json"
                    os.makedirs("conversations", exist_ok=True)
                    conversation_system.save_conversation(conversation_path)
                    last_conversation_save = current_time
                
            except Exception as e:
                logger.log_error(e, "Error in learning cycle")
                print(f"\nError in learning cycle: {e}")
                continue
                
    except KeyboardInterrupt:
        print(f"\nMother: Goodnight my dear child. (Age: {child.age()} months)")
        # Save final conversation
        final_conversation_path = f"conversations/final_conversation_{child.age()}mo.json"
        os.makedirs("conversations", exist_ok=True)
        conversation_system.save_conversation(final_conversation_path)
        
        # Final save on exit
        save_path = f"checkpoints/digital_child_{child.age()}mo_final"
        os.makedirs(save_path, exist_ok=True)
        
        # Save final state
        torch.save({
            'model_state': child.brain.state_dict(),
            'telemetry': telemetry,
            'curriculum_state': child.curriculum.current_stage,
            'emotional_state': child.emotional_state,
            'timestamp': time.time()
        }, f"{save_path}/model.pth")
        
        # Log final state
        logger.log_development("Final state saved")
        logger.cleanup()
        
        print(f"Final state saved: {save_path}")

if __name__ == "__main__":
    main()
