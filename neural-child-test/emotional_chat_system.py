# emotional_chat_system.py
# Description: Emotional chat system integrating heartbeat responses and memory recording
# Created by: Christopher Celaya

import torch
from datetime import datetime
from typing import Dict, Optional, List, Any
import json
from pathlib import Path
import re
import numpy as np

class EmotionalChatSystem:
    def __init__(self, brain, obsidian_api):
        """Initialize the emotional chat system.
        
        Args:
            brain: IntegratedBrain instance
            obsidian_api: ObsidianAPI instance
        """
        self.brain = brain
        self.obsidian_api = obsidian_api
        self.chat_history = []
        self.memory_commands = {
            '!remember': self._handle_remember_command,
            '!forget': self._handle_forget_command,
            '!reflect': self._handle_reflect_command
        }
        
    def process_message(self, message: str, tone: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Process an incoming chat message and update emotional state.
        
        Args:
            message (str): The incoming message
            tone (Dict[str, float], optional): Emotional tone analysis of the message
            
        Returns:
            Dict containing response and state information
        """
        # Check for memory commands
        if message.startswith('!'):
            command = message.split()[0]
            if command in self.memory_commands:
                return self.memory_commands[command](message)
        
        # Process message tone if not provided
        if not tone:
            tone = self._analyze_message_tone(message)
            
        # Update brain's emotional state
        emotional_response = self.brain.process_emotions(
            torch.tensor([0.0]),  # Placeholder for features
            torch.tensor([
                tone.get('joy', 0.0),
                tone.get('trust', 0.0),
                tone.get('fear', 0.0),
                tone.get('surprise', 0.0)
            ])
        )
        
        # Get current heartbeat state
        heartbeat_info = emotional_response.get('heartbeat', {})
        
        # Record interaction in chat history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'tone': tone,
            'heartbeat': heartbeat_info,
            'brain_state': self.brain.get_brain_state()
        }
        self.chat_history.append(chat_entry)
        
        # Record significant emotional interactions in Obsidian
        if self._is_significant_interaction(tone, heartbeat_info):
            self._record_in_obsidian(chat_entry)
            
        return {
            'emotional_response': emotional_response,
            'heartbeat': heartbeat_info,
            'brain_state': self.brain.get_brain_state()
        }
        
    def _analyze_message_tone(self, message: str) -> Dict[str, float]:
        """Analyze the emotional tone of a message.
        
        Args:
            message (str): The message to analyze
            
        Returns:
            Dict containing emotional tone values
        """
        # Simple rule-based tone analysis
        tone = {
            'joy': 0.0,
            'trust': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'anger': 0.0,
            'sadness': 0.0
        }
        
        # Positive indicators
        if re.search(r'happy|great|wonderful|love|excellent|:)', message.lower()):
            tone['joy'] = 0.8
            tone['trust'] = 0.6
            
        # Negative indicators
        if re.search(r'sad|angry|hate|terrible|awful|>:(', message.lower()):
            tone['anger'] = 0.8
            tone['sadness'] = 0.6
            
        # Fear/anxiety indicators
        if re.search(r'scared|afraid|worried|anxious|nervous', message.lower()):
            tone['fear'] = 0.8
            
        # Surprise indicators
        if re.search(r'wow|omg|amazing|incredible|unexpected|\!+', message.lower()):
            tone['surprise'] = 0.8
            
        return tone
        
    def _is_significant_interaction(self, tone: Dict[str, float], 
                                  heartbeat: Dict[str, Any]) -> bool:
        """Determine if an interaction is significant enough to record.
        
        Args:
            tone (Dict[str, float]): Message tone analysis
            heartbeat (Dict[str, Any]): Current heartbeat state
            
        Returns:
            bool: Whether the interaction is significant
        """
        # Check for high emotional values
        high_emotion = any(value > 0.7 for value in tone.values())
        
        # Check for significant heartbeat change
        significant_heartbeat = (
            heartbeat.get('state') in ['ANXIOUS', 'ELEVATED'] or
            abs(heartbeat.get('rate_change', 0)) > 10
        )
        
        return high_emotion or significant_heartbeat
        
    def _record_in_obsidian(self, chat_entry: Dict[str, Any]):
        """Record a significant interaction in Obsidian.
        
        Args:
            chat_entry (Dict[str, Any]): The chat interaction to record
        """
        # Create markdown content
        timestamp = datetime.fromisoformat(chat_entry['timestamp'])
        filename = f"memory_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        
        content = f"""# Memory Entry - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Interaction
Message: {chat_entry['message']}

## Emotional State
{json.dumps(chat_entry['tone'], indent=2)}

## Heartbeat Response
- Rate: {chat_entry['heartbeat'].get('current_rate', 'N/A')} BPM
- State: {chat_entry['heartbeat'].get('state', 'N/A')}
- Impact: {chat_entry['heartbeat'].get('emotional_impact', 'N/A')}

## Brain State
{json.dumps(chat_entry['brain_state'], indent=2)}

## Tags
#emotional-memory #heartbeat-response"""

        # Save to Obsidian vault
        self.obsidian_api.create_note(filename, content)
        
    def _handle_remember_command(self, message: str) -> Dict[str, Any]:
        """Handle !remember command to store explicit memories.
        
        Args:
            message (str): The full command message
            
        Returns:
            Dict containing command response
        """
        memory_content = message.replace('!remember', '').strip()
        
        # Create memory entry
        timestamp = datetime.now()
        filename = f"explicit_memory_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        
        content = f"""# Explicit Memory - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Content
{memory_content}

## Context
- Developmental Stage: {self.brain.stage.name}
- Brain State: {json.dumps(self.brain.get_brain_state(), indent=2)}

## Tags
#explicit-memory #commanded-memory"""

        # Save to Obsidian
        self.obsidian_api.create_note(filename, content)
        
        # Trigger memory processing in brain
        memory_response = self.brain.process_memory(
            torch.tensor([0.0]),  # Placeholder
            memory_type='explicit'
        )
        
        return {
            'status': 'Memory recorded',
            'memory_response': memory_response
        }
        
    def _handle_forget_command(self, message: str) -> Dict[str, Any]:
        """Handle !forget command to mark memories as forgotten.
        
        Args:
            message (str): The full command message
            
        Returns:
            Dict containing command response
        """
        # Implementation for forgetting memories
        # This could involve marking memories in Obsidian with a #forgotten tag
        return {'status': 'Not implemented yet'}
        
    def _handle_reflect_command(self, message: str) -> Dict[str, Any]:
        """Handle !reflect command to analyze emotional patterns.
        
        Args:
            message (str): The full command message
            
        Returns:
            Dict containing reflection analysis
        """
        # Get recent heartbeat history
        history = self.brain.heartbeat.get_heartbeat_history()
        
        # Analyze patterns
        analysis = self._analyze_emotional_patterns(history)
        
        # Record reflection in Obsidian
        timestamp = datetime.now()
        filename = f"reflection_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        
        content = f"""# Emotional Reflection - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Analysis
{json.dumps(analysis, indent=2)}

## Recent History
{json.dumps(history[-5:], indent=2)}

## Tags
#reflection #emotional-analysis"""

        self.obsidian_api.create_note(filename, content)
        
        return {
            'status': 'Reflection recorded',
            'analysis': analysis
        }
        
    def _analyze_emotional_patterns(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional patterns from heartbeat history.
        
        Args:
            history (List[Dict]): List of heartbeat history entries
            
        Returns:
            Dict containing pattern analysis
        """
        if not history:
            return {'status': 'No history available'}
            
        # Calculate statistics
        rates = [entry['rate'] for entry in history]
        states = [entry['state'] for entry in history]
        
        analysis = {
            'average_rate': sum(rates) / len(rates),
            'rate_variance': np.var(rates),
            'most_common_state': max(set(states), key=states.count),
            'state_transitions': len([i for i in range(1, len(states)) 
                                   if states[i] != states[i-1]]),
            'period_analyzed': f"{len(history)} entries"
        }
        
        return analysis 