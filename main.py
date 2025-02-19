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
from visualization import DevelopmentVisualizer

class MotherLLM:
    def __init__(self):
        self.emotional_history = []
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
        self.conversation_history.append({"role": "child", "content": child_response})
        prompt_text = self.stage_prompts.get(stage, self.stage_prompts[DevelopmentalStage.NEWBORN])
        prompt = prompt_text + f" Conversation history: {self.conversation_history}"
        response = chat_completion(
            system_prompt=prompt,
            user_prompt="Provide nurturing feedback and guidance:",
            structured_output=True
        )
        emotional_vector = torch.tensor([
            response.get('joy', 0.5),
            response.get('trust', 0.5),
            response.get('fear', 0.1),
            response.get('surprise', 0.3)
        ], device='cuda')
        if self.emotional_history:
            decay = 0.9 ** len(self.emotional_history)
            emotional_vector += self.emotional_history[-1] * decay
        self.emotional_history.append(emotional_vector)
        self.conversation_history.append({"role": "mother", "content": response['content']})
        self.feedback_history.append(response)
        return {
            'text': response['content'],
            'emotional_vector': torch.sigmoid(emotional_vector)
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
        return torch.tensor(
            get_embeddings(stimulus['text'])['data'][0]['embedding'], 
            device='cuda'
        ).unsqueeze(0)
        
    def respond(self, perception):
        with torch.amp.autocast("cuda"):
            return self.brain(perception)
    
    def learn(self, mother_feedback):
        self.memory.record_experience(
            mother_feedback['input'],
            mother_feedback['internal_state'],
            mother_feedback['reward'],
            time.time(),
            self.emotional_state  
        )
        return self.trainer.training_step(mother_feedback['input'])

    def age(self):
        return (datetime.now() - self.birth_date).days // 30  # months

def main():
    # Initialize logger and visualizer
    logger = DevelopmentLogger()
    visualizer = DevelopmentVisualizer()
    visualizer.initialize_plots()
    
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
        'memory_consolidation': []
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
    last_save_time = time.time()
    last_memory_check = time.time()
    last_visualization_update = time.time()
    
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
            except Exception as e:
                logger.log_error(e, "Error generating stimulus")
                print(f"\nError generating stimulus: {e}")
                time.sleep(5)  # Brief pause before retry
                continue
                
            # Update emotional state with error handling
            try:
                child.update_emotions(stimulus['emotional_vector'])
                emotional_stability = child.emotional_state.mean().item()
                telemetry['emotional_stability'].append(emotional_stability)
                
                # Log significant emotional events
                if abs(emotional_stability - np.mean(telemetry['emotional_stability'][-10:])) > 0.2:
                    logger.log_emotional_event(
                        child.express_feeling(),
                        emotional_stability,
                        str(stimulus['text'])
                    )
                    
            except Exception as e:
                logger.log_error(e, "Error in emotional update")
                print(f"\nError in emotional update: {e}")
                continue
            
            # Process perception and response
            try:
                perception = child.perceive(stimulus)
                response = child.respond(perception)
                
                # Get feedback from mother
                feedback = chat_completion(
                    system_prompt=f"Evaluate this response from your digital child: {response}",
                    user_prompt=f"Child expressed {child.express_feeling()}. Provide nurturing feedback:",
                    structured_output=True
                )
                
                # Ensure feedback has all required fields
                if feedback is None or not isinstance(feedback, dict):
                    logger.log_error("Invalid feedback format", "Mother feedback was not in correct format")
                    feedback = {
                        'reward_score': 0.5,
                        'success_metric': 0.5,
                        'complexity_rating': 0.5,
                        'self_critique_score': 0.5
                    }
                else:
                    # Ensure all required fields exist with default values if missing
                    default_values = {
                        'reward_score': 0.5,
                        'success_metric': 0.5,
                        'complexity_rating': 0.5,
                        'self_critique_score': 0.5
                    }
                    for key, default_value in default_values.items():
                        if key not in feedback or feedback[key] is None:
                            feedback[key] = default_value
                
                # Log interaction with error handling
                try:
                    logger.log_interaction(stimulus, response, feedback)
                except Exception as e:
                    logger.log_error(e, "Failed to log interaction")
                
                # Learning step with error handling
                try:
                    loss = child.learn({
                        'input': perception,
                        'internal_state': response,
                        'reward': feedback['reward_score']
                    })
                except Exception as e:
                    logger.log_error(e, "Failed in learning step")
                    loss = 0.0
                    continue
                
                # Update telemetry
                telemetry['loss'].append(loss)
                telemetry['memory_usage'].append(memory_usage)
                telemetry['moral_scores'].append(feedback['reward_score'])
                
                # Update psychological metrics
                telemetry['psychological']['attachment_style'].append(
                    child.brain.attachment.attachment_styles.tolist()
                )
                telemetry['psychological']['defense_mechanisms'].append(
                    child.brain.defense_mechanisms.anxiety_threshold.item()
                )
                telemetry['psychological']['theory_of_mind'].append(
                    child.brain.theory_of_mind.social_bias.tolist()
                )
                
                # Log performance metrics
                logger.log_performance({
                    'loss': loss,
                    'memory_usage': memory_usage,
                    'moral_score': feedback['reward_score'],
                    'emotional_stability': emotional_stability
                })
                
                # Stage progression check
                old_stage = child.curriculum.current_stage
                stage_update = child.curriculum.update_stage({
                    'success_rate': feedback['success_metric'],
                    'abstraction': feedback['complexity_rating'],
                    'self_awareness': feedback['self_critique_score']
                })
                
                if stage_update:
                    print(f"\n{stage_update}")
                    telemetry['stage_transitions'].append({
                        'time': current_time,
                        'stage': child.curriculum.current_stage.name,
                        'message': stage_update
                    })
                    
                    # Log stage transition
                    logger.log_stage_transition(
                        old_stage.name,
                        child.curriculum.current_stage.name,
                        {
                            'success_rate': feedback['success_metric'],
                            'abstraction': feedback['complexity_rating'],
                            'self_awareness': feedback['self_critique_score']
                        }
                    )
                
                # Update visualization
                if current_time - last_visualization_update > 5:  # Update every 5 seconds
                    visualizer.update_plots(telemetry)
                    last_visualization_update = current_time
                
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
                    
                    # Save visualizations
                    visualizer.save_visualization(telemetry, save_path)
                    
                    # Generate and save development report
                    report = visualizer.generate_report(telemetry, save_path)
                    logger.log_development(f"Checkpoint saved: {save_path}")
                    logger.log_development(f"Development Report: {json.dumps(report, indent=2)}")
                    
                    last_save_time = current_time
                    print(f"\nCheckpoint saved: {save_path}")
                
            except Exception as e:
                logger.log_error(e, "Error in learning cycle")
                print(f"\nError in learning cycle: {e}")
                continue
                
    except KeyboardInterrupt:
        print(f"\nMother: Goodnight my dear child. (Age: {child.age()} months)")
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
        
        # Save final visualizations and report
        visualizer.save_visualization(telemetry, save_path)
        report = visualizer.generate_report(telemetry, save_path)
        
        # Log final state
        logger.log_development("Final state saved")
        logger.log_development(f"Final Development Report: {json.dumps(report, indent=2)}")
        logger.cleanup()
        
        print(f"Final state saved: {save_path}")

if __name__ == "__main__":
    main()
