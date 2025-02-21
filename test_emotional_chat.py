"""
test_emotional_chat.py
Description: Test suite for the emotional chat system
Created by: Christopher Celaya
"""

import unittest
from datetime import datetime, timedelta
import torch
import numpy as np
from emotional_memory_system import EmotionalMemorySystem, EmotionalMemoryEntry
from integrated_brain import IntegratedBrain
from heartbeat_system import HeartbeatSystem

class TestEmotionalChat(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.brain = IntegratedBrain()
        self.heartbeat = HeartbeatSystem()
        self.emotional_system = EmotionalMemorySystem(self.brain)
        
    def test_emotional_memory_creation(self):
        """Test creating and storing emotional memories"""
        # Create a test memory
        memory = EmotionalMemoryEntry(
            content="I feel happy today!",
            emotional_state={
                'joy': 0.8,
                'trust': 0.7,
                'fear': 0.1,
                'surprise': 0.3
            },
            context="Testing emotional memory",
            intensity=0.7,
            valence=0.8,
            arousal=0.6,
            timestamp=datetime.now(),
            metadata={'test': True}
        )
        
        # Store memory
        memory_id = self.emotional_system.store_memory(memory)
        self.assertIsNotNone(memory_id)
        
        # Retrieve similar memories
        similar = self.emotional_system.retrieve_similar_memories("happy")
        self.assertTrue(len(similar) > 0)
        
    def test_emotional_pattern_analysis(self):
        """Test analyzing emotional patterns"""
        # Create test heartbeat history
        history = [
            {'rate': 80, 'state': 'RESTING'},
            {'rate': 90, 'state': 'EXCITED'},
            {'rate': 85, 'state': 'RESTING'},
            {'rate': 75, 'state': 'CALM'}
        ]
        
        # Analyze patterns
        analysis = self.emotional_system._analyze_emotional_patterns(history)
        
        self.assertIn('average_rate', analysis)
        self.assertIn('rate_stability', analysis)
        self.assertIn('emotional_resilience', analysis)
        
    def test_memory_consolidation(self):
        """Test memory consolidation process"""
        # Create test memories
        memories = [
            EmotionalMemoryEntry(
                content="Happy memory",
                emotional_state={'joy': 0.9, 'trust': 0.8, 'fear': 0.1, 'surprise': 0.2},
                context="Test",
                intensity=0.8,
                valence=0.9,
                arousal=0.5,
                timestamp=datetime.now(),
                metadata={}
            ),
            EmotionalMemoryEntry(
                content="Scary memory",
                emotional_state={'joy': 0.1, 'trust': 0.2, 'fear': 0.9, 'surprise': 0.8},
                context="Test",
                intensity=0.9,
                valence=-0.8,
                arousal=0.9,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        # Store memories
        for memory in memories:
            self.emotional_system.store_memory(memory)
            
        # Test consolidation
        result = self.emotional_system.consolidate_memories()
        self.assertEqual(result['status'], 'Memory consolidation complete')
        self.assertIn('metrics', result)
        
    def test_emotional_state_updates(self):
        """Test emotional state updates from consolidation"""
        # Create initial brain state
        self.brain.brain_state.emotional_valence = 0.5
        self.brain.brain_state.arousal = 0.3
        
        # Create test memory groups
        memory_groups = {
            'positive': [
                EmotionalMemoryEntry(
                    content="Very happy",
                    emotional_state={'joy': 0.9, 'trust': 0.8, 'fear': 0.1, 'surprise': 0.2},
                    context="Test",
                    intensity=0.8,
                    valence=0.9,
                    arousal=0.5,
                    timestamp=datetime.now(),
                    metadata={}
                )
            ],
            'negative': [],
            'neutral': [],
            'complex': [],
            'traumatic': []
        }
        
        # Create test metrics
        metrics = {
            'total_memories': 1,
            'positive_ratio': 1.0,
            'negative_ratio': 0.0,
            'emotional_complexity': 0.0,
            'trauma_exposure': 0.0
        }
        
        # Update emotional state
        self.emotional_system._update_emotional_state_from_consolidation(memory_groups, metrics)
        
        # Check if emotional state was updated
        self.assertGreater(self.brain.brain_state.emotional_valence, 0.5)
        self.assertIsInstance(self.brain.brain_state.arousal, float)
        
if __name__ == '__main__':
    unittest.main() 