# tests/test_rag_memory.py
# Description: Tests for the RAG memory system
# Created by: Christopher Celaya

import unittest
import torch
import numpy as np
from datetime import datetime
from rag_memory import RAGMemorySystem, MemoryContext

class TestRAGMemory(unittest.TestCase):
    def setUp(self):
        self.memory_system = RAGMemorySystem(persist_dir="test_rag_memories")
        self.test_emotional_state = {
            'joy': 0.7,
            'trust': 0.6,
            'fear': 0.2,
            'surprise': 0.3
        }
        self.test_brain_state = {
            'arousal': 0.6,
            'attention': 0.7,
            'emotional_valence': 0.5,
            'consciousness': 0.9
        }
        
    def tearDown(self):
        # Clear test memories
        self.memory_system.clear_memories()
        
    def test_store_memory(self):
        """Test storing memories of different types"""
        # Store emotional memory
        emotional_id = self.memory_system.store_memory(
            content="I felt happy playing with my toys",
            memory_type="emotional",
            emotional_state=self.test_emotional_state,
            brain_state=self.test_brain_state,
            metadata={
                'age_months': 12,
                'developmental_stage': 'EARLY_TODDLER'
            }
        )
        self.assertIsNotNone(emotional_id)
        
        # Store episodic memory
        episodic_id = self.memory_system.store_memory(
            content="Today I learned to stack blocks",
            memory_type="episodic",
            emotional_state=self.test_emotional_state,
            brain_state=self.test_brain_state,
            metadata={
                'age_months': 12,
                'developmental_stage': 'EARLY_TODDLER'
            }
        )
        self.assertIsNotNone(episodic_id)
        
        # Store semantic memory
        semantic_id = self.memory_system.store_memory(
            content="A block is a toy you can stack",
            memory_type="semantic",
            emotional_state=self.test_emotional_state,
            brain_state=self.test_brain_state,
            metadata={
                'age_months': 12,
                'developmental_stage': 'EARLY_TODDLER'
            }
        )
        self.assertIsNotNone(semantic_id)
        
        # Check stats
        stats = self.memory_system.get_stats()
        self.assertEqual(stats['total_memories'], 3)
        self.assertEqual(stats['emotional_memories'], 1)
        self.assertEqual(stats['episodic_memories'], 1)
        self.assertEqual(stats['semantic_memories'], 1)
        
    def test_retrieve_memories(self):
        """Test retrieving memories with context"""
        # Store some test memories
        self.memory_system.store_memory(
            content="I felt happy playing with my red ball",
            memory_type="emotional",
            emotional_state={'joy': 0.8, 'trust': 0.7, 'fear': 0.1, 'surprise': 0.2},
            brain_state=self.test_brain_state,
            metadata={'age_months': 12, 'developmental_stage': 'EARLY_TODDLER'}
        )
        
        self.memory_system.store_memory(
            content="I was scared of the loud noise",
            memory_type="emotional",
            emotional_state={'joy': 0.2, 'trust': 0.3, 'fear': 0.8, 'surprise': 0.7},
            brain_state=self.test_brain_state,
            metadata={'age_months': 12, 'developmental_stage': 'EARLY_TODDLER'}
        )
        
        self.memory_system.store_memory(
            content="Playing with blocks helps develop motor skills",
            memory_type="semantic",
            emotional_state=self.test_emotional_state,
            brain_state=self.test_brain_state,
            metadata={'age_months': 12, 'developmental_stage': 'EARLY_TODDLER'}
        )
        
        # Create retrieval context
        context = MemoryContext(
            query="playing with toys",
            emotional_state={'joy': 0.7, 'trust': 0.6, 'fear': 0.2, 'surprise': 0.3},
            brain_state=self.test_brain_state,
            developmental_stage='EARLY_TODDLER',
            age_months=12,
            timestamp=datetime.now()
        )
        
        # Retrieve memories
        memories = self.memory_system.retrieve_memories(
            context=context,
            memory_types=['emotional', 'semantic'],
            n_results=5
        )
        
        # Verify results
        self.assertGreater(len(memories), 0)
        self.assertIn('relevance', memories[0])
        self.assertIn('emotional_similarity', memories[0])
        self.assertIn('temporal_relevance', memories[0])
        self.assertIn('developmental_relevance', memories[0])
        
        # Check that most relevant memory is about playing with ball
        self.assertIn('ball', memories[0]['content'].lower())
        
    def test_emotional_similarity(self):
        """Test emotional similarity calculation"""
        state1 = {'joy': 0.8, 'trust': 0.7, 'fear': 0.1, 'surprise': 0.2}
        state2 = {'joy': 0.7, 'trust': 0.6, 'fear': 0.2, 'surprise': 0.3}
        state3 = {'joy': 0.2, 'trust': 0.3, 'fear': 0.8, 'surprise': 0.7}
        
        sim1 = self.memory_system._calculate_emotional_similarity(state1, state2)
        sim2 = self.memory_system._calculate_emotional_similarity(state1, state3)
        
        # Similar states should have high similarity
        self.assertGreater(sim1, 0.8)
        # Opposite states should have low similarity
        self.assertLess(sim2, 0.5)
        
    def test_temporal_relevance(self):
        """Test temporal relevance calculation"""
        now = datetime.now()
        old_timestamp = datetime(2023, 1, 1)
        recent_timestamp = now
        
        old_relevance = self.memory_system._calculate_temporal_relevance(old_timestamp)
        recent_relevance = self.memory_system._calculate_temporal_relevance(recent_timestamp)
        
        # Recent memories should be more relevant
        self.assertGreater(recent_relevance, old_relevance)
        self.assertGreater(recent_relevance, 0.9)  # Very recent = high relevance
        
    def test_developmental_relevance(self):
        """Test developmental relevance calculation"""
        # Same stage and age
        relevance1 = self.memory_system._calculate_developmental_relevance(
            current_age=12,
            memory_age=12,
            current_stage='EARLY_TODDLER',
            memory_stage='EARLY_TODDLER'
        )
        
        # Different stage but close age
        relevance2 = self.memory_system._calculate_developmental_relevance(
            current_age=12,
            memory_age=11,
            current_stage='EARLY_TODDLER',
            memory_stage='INFANT'
        )
        
        # Different stage and distant age
        relevance3 = self.memory_system._calculate_developmental_relevance(
            current_age=12,
            memory_age=3,
            current_stage='EARLY_TODDLER',
            memory_stage='NEWBORN'
        )
        
        # Same stage and age should be most relevant
        self.assertGreater(relevance1, relevance2)
        self.assertGreater(relevance2, relevance3)
        self.assertEqual(relevance1, 1.0)  # Perfect relevance
        
    def test_memory_integration(self):
        """Test integration of different memory types"""
        # Store memories of different types with related content
        self.memory_system.store_memory(
            content="I felt happy when I learned about colors",
            memory_type="emotional",
            emotional_state={'joy': 0.8, 'trust': 0.7, 'fear': 0.1, 'surprise': 0.6},
            brain_state=self.test_brain_state,
            metadata={'age_months': 12, 'developmental_stage': 'EARLY_TODDLER'}
        )
        
        self.memory_system.store_memory(
            content="Today I learned that the ball is red",
            memory_type="episodic",
            emotional_state={'joy': 0.7, 'trust': 0.6, 'fear': 0.2, 'surprise': 0.5},
            brain_state=self.test_brain_state,
            metadata={'age_months': 12, 'developmental_stage': 'EARLY_TODDLER'}
        )
        
        self.memory_system.store_memory(
            content="Red is a color that we see in many toys",
            memory_type="semantic",
            emotional_state=self.test_emotional_state,
            brain_state=self.test_brain_state,
            metadata={'age_months': 12, 'developmental_stage': 'EARLY_TODDLER'}
        )
        
        # Create retrieval context about colors
        context = MemoryContext(
            query="learning about colors and red things",
            emotional_state={'joy': 0.7, 'trust': 0.6, 'fear': 0.2, 'surprise': 0.4},
            brain_state=self.test_brain_state,
            developmental_stage='EARLY_TODDLER',
            age_months=12,
            timestamp=datetime.now()
        )
        
        # Retrieve all memory types
        memories = self.memory_system.retrieve_memories(context=context)
        
        # Verify integration
        self.assertGreaterEqual(len(memories), 3)
        memory_types = [m['type'] for m in memories]
        self.assertIn('emotional', memory_types)
        self.assertIn('episodic', memory_types)
        self.assertIn('semantic', memory_types)
        
        # Check relevance ordering
        relevances = [m['relevance'] for m in memories]
        self.assertEqual(relevances, sorted(relevances, reverse=True))

if __name__ == '__main__':
    unittest.main() 