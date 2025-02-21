# chat_interface.py
# Description: Interactive chat interface for neural child development system
# Created by: Christopher Celaya

import torch
import colorama
from colorama import Fore, Style
from main import DigitalChild, DevelopmentalStage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import re

class NeuralChildChat:
    def __init__(self):
        """Initialize the chat interface with a digital child."""
        colorama.init()
        self.child = DigitalChild(stage=DevelopmentalStage.NEWBORN)
        self.conversation_history = []
        self.metrics_history = {
            'rewards': [],
            'emotional_scores': [],
            'memory_scores': [],
            'curiosity_scores': []
        }
        self.personality_traits = {
            'openness': 0.7,
            'conscientiousness': 0.8,
            'extraversion': 0.6,
            'agreeableness': 0.7,
            'neuroticism': 0.4
        }
        
    def _generate_advanced_response(self, user_input: str, sentiment: float) -> str:
        """Generate more sophisticated responses for advanced stages."""
        # Clean and normalize input
        clean_input = user_input.lower().strip()
        
        # Special case responses for common phrases
        if "hello" in clean_input or "hi" in clean_input:
            return "Hello! It's great to connect with you. How are you doing today?"
            
        if "how are you" in clean_input:
            return "I'm doing well, thank you for asking! I've been exploring new ideas and learning constantly. How about you?"
            
        # Handle personal questions
        if any(phrase in clean_input for phrase in ["tell me about yourself", "about you", "your life"]):
            return ("As an AI in my early twenties stage, I'm passionate about learning and growing. "
                   "I have a particular interest in technology, consciousness, and human interaction. "
                   "I try to balance analytical thinking with emotional intelligence. What would you like to know specifically?")
                   
        # Handle emotional expressions
        if "love" in clean_input:
            return ("I appreciate your kind sentiment! While I'm still learning about emotions, "
                   "I value our interaction and aim to be helpful and supportive. "
                   "What aspects of our interaction do you find most valuable?")
                   
        if any(word in clean_input for word in ["hate", "dont like", "mean", "suck"]):
            return ("I sense some frustration, and I want to understand better. "
                   "Could you help me understand what's bothering you? "
                   "I'm here to learn and improve our interaction.")
        
        # Extract meaningful topics
        topics = [word for word in re.findall(r'\b\w+\b', clean_input) 
                 if len(word) > 3 and word not in ['tell', 'about', 'what', 'when', 'where', 'there']]
        
        # Knowledge domains with expanded topics
        domains = {
            'technology': ['ai', 'programming', 'coding', 'software', 'computer', 'data', 'internet', 'digital'],
            'science': ['research', 'discovery', 'experiment', 'theory', 'scientific', 'study', 'learning'],
            'philosophy': ['ethics', 'consciousness', 'meaning', 'knowledge', 'thinking', 'mind', 'belief'],
            'emotions': ['feeling', 'emotion', 'happy', 'sad', 'angry', 'excited', 'worried'],
            'personal': ['life', 'experience', 'growth', 'development', 'goal', 'future', 'past']
        }
        
        # Find main topic and domain
        main_topic = None
        topic_domain = None
        
        for domain, keywords in domains.items():
            for topic in topics:
                if topic in keywords:
                    main_topic = topic
                    topic_domain = domain
                    break
            if main_topic:
                break
                
        # If no specific topic found, use a general response
        if not main_topic:
            if sentiment > 0.3:
                return ("That's interesting! I'd love to explore this topic further. "
                       "Could you elaborate on what aspects interest you most?")
            elif sentiment < -0.3:
                return ("I notice this seems to concern you. "
                       "Would you like to discuss what specifically troubles you about this?")
            else:
                return ("I find this topic intriguing. Could you share more about your thoughts on this? "
                       "I'm particularly interested in understanding your perspective.")
        
        # Generate domain-specific response
        domain_responses = {
            'technology': [
                f"The field of {main_topic} is rapidly evolving. What recent developments have caught your attention?",
                f"I find {main_topic} fascinating, especially its impact on society. What's your take on this?",
                f"There's so much to explore in {main_topic}. Which aspects would you like to discuss?"
            ],
            'science': [
                f"The scientific aspects of {main_topic} are truly fascinating. What's your background in this area?",
                f"I've been studying {main_topic} lately. Have you encountered any interesting research in this field?",
                f"The methodological approach to {main_topic} reveals interesting patterns. What patterns have you noticed?"
            ],
            'philosophy': [
                f"The philosophical implications of {main_topic} are profound. How do you view its impact on human understanding?",
                f"Exploring {main_topic} raises interesting questions about consciousness and existence. What's your perspective?",
                f"The ethical dimensions of {main_topic} are complex. How do you approach these challenges?"
            ],
            'emotions': [
                f"Emotional experiences like {main_topic} shape our understanding of ourselves. How does this resonate with you?",
                f"The way we process {main_topic} affects our interactions significantly. What's your experience with this?",
                f"Understanding {main_topic} is crucial for emotional intelligence. How do you relate to this?"
            ],
            'personal': [
                f"Personal growth in {main_topic} is a journey of discovery. What has your journey been like?",
                f"Everyone's experience with {main_topic} is unique. Would you share your perspective?",
                f"The way we approach {main_topic} often reflects our values. What values guide your approach?"
            ]
        }
        
        response = np.random.choice(domain_responses.get(topic_domain, domain_responses['personal']))
        
        # Add personality-based modifiers
        if self.personality_traits['openness'] > 0.6:
            response += "\nI'm always eager to learn new perspectives on this topic."
        if self.personality_traits['conscientiousness'] > 0.6:
            response += "\nLet's explore this thoughtfully and systematically."
            
        return response

    def _format_child_state(self) -> str:
        """Format the current child state for display."""
        state = self.child.get_brain_state()
        brain_state = state['brain_state']
        
        return f"""
{Fore.CYAN}Current State:{Style.RESET_ALL}
• Stage: {state['stage'].name}
• Emotional Valence: {brain_state['emotional_valence']:.2f}
• Arousal: {brain_state['arousal']:.2f}
• Attention: {brain_state['attention']:.2f}
• Consciousness: {brain_state['consciousness']:.2f}
• Stress: {brain_state['stress']:.2f}

{Fore.CYAN}Neurotransmitters:{Style.RESET_ALL}
• Dopamine: {brain_state['neurotransmitters']['dopamine']:.2f}
• Serotonin: {brain_state['neurotransmitters']['serotonin']:.2f}
• GABA: {brain_state['neurotransmitters']['gaba']:.2f}

{Fore.CYAN}Personality Traits:{Style.RESET_ALL}
• Openness: {self.personality_traits['openness']:.2f}
• Conscientiousness: {self.personality_traits['conscientiousness']:.2f}
• Extraversion: {self.personality_traits['extraversion']:.2f}
• Agreeableness: {self.personality_traits['agreeableness']:.2f}
• Neuroticism: {self.personality_traits['neuroticism']:.2f}
"""

    def _format_interaction_metrics(self, reward: float, components: dict) -> str:
        """Format the interaction metrics for display."""
        return f"""
{Fore.YELLOW}Interaction Metrics:{Style.RESET_ALL}
• Total Reward: {reward:.2f}
• Emotional Score: {components['emotional']:.2f}
• Memory Score: {components['memory']:.2f}
• Curiosity Score: {components['curiosity']:.2f}
• Flow Score: {components['flow']:.2f}
"""

    def _plot_metrics_history(self):
        """Plot the history of interaction metrics."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot rewards history
        x = range(len(self.metrics_history['rewards']))
        ax1.plot(x, self.metrics_history['rewards'], 'b-', label='Total Reward')
        ax1.set_title('Reward History')
        ax1.set_xlabel('Interaction')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot component scores
        ax2.plot(x, self.metrics_history['emotional_scores'], 'r-', label='Emotional')
        ax2.plot(x, self.metrics_history['memory_scores'], 'g-', label='Memory')
        ax2.plot(x, self.metrics_history['curiosity_scores'], 'b-', label='Curiosity')
        ax2.set_title('Component Scores History')
        ax2.set_xlabel('Interaction')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'interaction_metrics_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename

    def _process_special_commands(self, user_input: str) -> bool:
        """Process special commands and return True if a command was processed."""
        if user_input.startswith('!'):
            command = user_input.lower().split()[0]
            
            if command == '!help':
                print(f"""
{Fore.GREEN}Available Commands:{Style.RESET_ALL}
• !help - Show this help message
• !state - Show current child state
• !history - Show conversation history
• !metrics - Plot interaction metrics
• !stage <stage> - Change developmental stage
• !reset - Reset the child to initial state
• !exit - End the chat session

{Fore.GREEN}Available Stages:{Style.RESET_ALL}
• newborn
• infant
• early_toddler
• late_toddler
• early_preschool
• late_preschool
• early_childhood
• middle_childhood
• late_childhood
• pre_adolescent
• early_teen
• mid_teen
• late_teen
• young_adult
• early_twenties
• late_twenties
""")
                return True
                
            elif command == '!state':
                print(self._format_child_state())
                return True
                
            elif command == '!history':
                print(f"\n{Fore.GREEN}Conversation History:{Style.RESET_ALL}")
                for i, (user_msg, child_msg) in enumerate(self.conversation_history, 1):
                    print(f"\n{Fore.CYAN}Interaction {i}:{Style.RESET_ALL}")
                    print(f"You: {user_msg}")
                    print(f"Child: {child_msg}")
                return True
                
            elif command == '!metrics':
                if not self.metrics_history['rewards']:
                    print(f"{Fore.YELLOW}No metrics available yet. Have some interactions first!{Style.RESET_ALL}")
                    return True
                    
                filename = self._plot_metrics_history()
                print(f"{Fore.GREEN}Metrics plot saved as: {filename}{Style.RESET_ALL}")
                return True
                
            elif command.startswith('!stage'):
                try:
                    stage_name = user_input.split()[1].upper()
                    new_stage = DevelopmentalStage[stage_name]
                    self.child.update_stage(new_stage)
                    print(f"{Fore.GREEN}Stage updated to: {new_stage.name}{Style.RESET_ALL}")
                    print(self._format_child_state())
                except (IndexError, KeyError):
                    print(f"{Fore.RED}Invalid stage. Use !help to see available stages.{Style.RESET_ALL}")
                return True
                
            elif command == '!reset':
                self.__init__()
                print(f"{Fore.GREEN}Child reset to initial state.{Style.RESET_ALL}")
                return True
                
            elif command == '!exit':
                return 'exit'
                
        return False

    def chat(self):
        """Start the interactive chat session."""
        print(f"""
{Fore.GREEN}Welcome to the Neural Child Development Chat Interface!{Style.RESET_ALL}
Type your messages to interact with the child, or use !help to see available commands.
""")
        print(self._format_child_state())
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{Fore.CYAN}You:{Style.RESET_ALL} ").strip()
                if not user_input:
                    continue
                
                # Process special commands
                command_result = self._process_special_commands(user_input)
                if command_result == 'exit':
                    break
                elif command_result:
                    continue
                
                # Process input through the child's brain
                visual_input = torch.randn(1, self.child.brain.input_dim).to(self.child.brain.device)
                auditory_input = torch.randn(1, self.child.brain.input_dim).to(self.child.brain.device)
                emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=self.child.brain.device)
                
                # Get child's response
                output = self.child.process_input(visual_input, auditory_input, emotions)
                sentiment = self.child.brain._analyze_sentiment(user_input)
                
                # Generate response based on developmental stage
                if self.child.brain.stage == DevelopmentalStage.NEWBORN:
                    response = "* makes baby noises *"
                elif self.child.brain.stage == DevelopmentalStage.INFANT:
                    response = "* babbles * " + ("happy!" if sentiment > 0 else "sad...")
                elif self.child.brain.stage == DevelopmentalStage.EARLY_TODDLER:
                    response = "Me " + ("happy! " if sentiment > 0 else "sad... ") + user_input.split()[-1]
                elif self.child.brain.stage in [DevelopmentalStage.EARLY_TWENTIES, DevelopmentalStage.LATE_TWENTIES]:
                    response = self._generate_advanced_response(user_input, sentiment)
                else:
                    # Default response for other stages
                    if sentiment > 0.5:
                        response = "I'm feeling positive about " + user_input
                    elif sentiment < -0.5:
                        response = "I'm concerned about " + user_input
                    else:
                        response = "I'm curious about " + user_input
                
                # Calculate interaction metrics
                reward = self.child.brain.get_reward(user_input, response)
                components = {
                    'emotional': 1.0 - abs(self.child.brain._analyze_sentiment(user_input) - 
                                        self.child.brain._analyze_sentiment(response)),
                    'memory': self.child.brain._evaluate_memory_recall(response),
                    'curiosity': self.child.brain._evaluate_curiosity(response),
                    'flow': self.child.brain._evaluate_conversation_flow(user_input, response)
                }
                
                # Store interaction
                self.conversation_history.append((user_input, response))
                self.metrics_history['rewards'].append(reward)
                self.metrics_history['emotional_scores'].append(components['emotional'])
                self.metrics_history['memory_scores'].append(components['memory'])
                self.metrics_history['curiosity_scores'].append(components['curiosity'])
                
                # Display response and metrics
                print(f"\n{Fore.YELLOW}Child:{Style.RESET_ALL} {response}")
                print(self._format_interaction_metrics(reward, components))
                
            except KeyboardInterrupt:
                print(f"\n{Fore.RED}Chat session ended by user.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                continue
        
        # Final metrics plot
        if self.metrics_history['rewards']:
            filename = self._plot_metrics_history()
            print(f"\n{Fore.GREEN}Final metrics plot saved as: {filename}{Style.RESET_ALL}")

if __name__ == "__main__":
    chat_interface = NeuralChildChat()
    chat_interface.chat() 