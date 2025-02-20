# conversation_system.py
# Description: Conversation system for neural child development
# Created by: Christopher Celaya

import json
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any
from logger import DevelopmentLogger

class ConversationSystem:
    def __init__(self, logger: DevelopmentLogger):
        """Initialize the conversation system."""
        self.logger = logger
        self.conversation_history = []
        self.current_context = {}
        self.last_response = None
        
    def process_mother_response(self, response: str, emotional_state: torch.Tensor) -> Dict:
        """Process mother's response and update emotional state."""
        try:
            # Log the interaction
            interaction = {
                'role': 'mother',
                'content': response,
                'emotional_state': emotional_state.cpu().tolist(),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_interaction(interaction)
            
            # Add to conversation history
            self.conversation_history.append(interaction)
            
            # Update context
            self.current_context.update({
                'last_mother_response': response,
                'last_emotional_state': emotional_state.cpu().tolist()
            })
            
            return {
                'success': True,
                'response': response,
                'emotional_impact': emotional_state.cpu().tolist()
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'process_mother_response',
                'response': response,
                'emotional_state': emotional_state.cpu().tolist() if emotional_state is not None else None
            })
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_child_response(self, response: str, emotional_state: torch.Tensor) -> Dict:
        """Process child's response and update conversation history."""
        try:
            # Log the interaction
            interaction = {
                'role': 'child',
                'content': response,
                'emotional_state': emotional_state.cpu().tolist(),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_interaction(interaction)
            
            # Add to conversation history
            self.conversation_history.append(interaction)
            
            # Update context
            self.current_context.update({
                'last_child_response': response,
                'last_emotional_state': emotional_state.cpu().tolist()
            })
            
            self.last_response = response
            
            return {
                'success': True,
                'response': response,
                'emotional_state': emotional_state.cpu().tolist()
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'process_child_response',
                'response': response,
                'emotional_state': emotional_state.cpu().tolist() if emotional_state is not None else None
            })
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_conversation_context(self, window_size: int = 5) -> List[Dict]:
        """Get recent conversation context."""
        return self.conversation_history[-window_size:] if self.conversation_history else []
    
    def get_emotional_trajectory(self) -> List[float]:
        """Get emotional state trajectory from conversation history."""
        emotional_states = []
        for interaction in self.conversation_history:
            if 'emotional_state' in interaction:
                emotional_states.append(interaction['emotional_state'])
        return emotional_states
    
    def save_conversation(self, filepath: str):
        """Save conversation history to a file."""
        try:
            data = {
                'history': self.conversation_history,
                'context': self.current_context,
                'timestamp': datetime.now().isoformat()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'save_conversation',
                'filepath': filepath
            })
    
    def load_conversation(self, filepath: str):
        """Load conversation history from a file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.conversation_history = data.get('history', [])
            self.current_context = data.get('context', {})
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'load_conversation',
                'filepath': filepath
            })
    
    def clear_history(self):
        """Clear conversation history and context."""
        self.conversation_history = []
        self.current_context = {}
        self.last_response = None
        
    def analyze_conversation(self) -> Dict[str, Any]:
        """Analyze the conversation history for patterns and metrics."""
        if not self.conversation_history:
            return {
                'total_interactions': 0,
                'mother_responses': 0,
                'child_responses': 0,
                'average_emotional_state': None,
                'conversation_duration': 0
            }
        
        try:
            # Calculate basic metrics
            total_interactions = len(self.conversation_history)
            mother_responses = sum(1 for x in self.conversation_history if x['role'] == 'mother')
            child_responses = sum(1 for x in self.conversation_history if x['role'] == 'child')
            
            # Calculate average emotional state
            emotional_states = [
                interaction['emotional_state']
                for interaction in self.conversation_history
                if 'emotional_state' in interaction
            ]
            average_emotional_state = (
                torch.tensor(emotional_states).mean(dim=0).tolist()
                if emotional_states else None
            )
            
            # Calculate conversation duration
            if len(self.conversation_history) >= 2:
                start_time = datetime.fromisoformat(self.conversation_history[0]['timestamp'])
                end_time = datetime.fromisoformat(self.conversation_history[-1]['timestamp'])
                duration = (end_time - start_time).total_seconds()
            else:
                duration = 0
            
            return {
                'total_interactions': total_interactions,
                'mother_responses': mother_responses,
                'child_responses': child_responses,
                'average_emotional_state': average_emotional_state,
                'conversation_duration': duration
            }
            
        except Exception as e:
            self.logger.log_error(str(e), {
                'method': 'analyze_conversation'
            })
            return {
                'error': str(e)
            } 