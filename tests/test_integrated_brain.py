# tests/test_integrated_brain.py
# Description: Tests for the integrated brain architecture
# Created by: Christopher Celaya

import unittest
import torch
import numpy as np
from main import IntegratedBrain, DigitalChild, DevelopmentalStage

class TestIntegratedBrain(unittest.TestCase):
    def setUp(self):
        self.brain = IntegratedBrain()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = 384
        self.batch_size = 1
        
    def test_sensory_processing(self):
        """Test sensory system processing"""
        # Create test inputs
        visual_input = torch.randn(self.batch_size, self.input_dim)
        auditory_input = torch.randn(self.batch_size, self.input_dim)
        
        # Process through sensory system
        sensory_output = self.brain.process_sensory_input(visual_input, auditory_input)
        
        # Verify output
        self.assertIn('sensory_features', sensory_output)
        self.assertEqual(sensory_output['sensory_features'].shape[0], self.batch_size)
        
    def test_memory_processing(self):
        """Test memory system processing"""
        # Create test input
        features = torch.randn(self.batch_size, self.brain.hidden_dim)
        
        # Process through memory system
        memory_output = self.brain.process_memory(features)
        
        # Verify outputs
        self.assertIn('working_memory', memory_output)
        self.assertIn('episodic_memory', memory_output)
        self.assertIn('semantic_memory', memory_output)
        self.assertIn('integrated_memory', memory_output)
        
    def test_emotional_processing(self):
        """Test emotional system processing"""
        # Create test inputs
        features = torch.randn(self.batch_size, self.brain.hidden_dim)
        current_emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]])  # [joy, trust, fear, surprise]
        
        # Process through emotional system
        emotional_output = self.brain.process_emotions(features, current_emotions)
        
        # Verify outputs
        self.assertIn('emotions', emotional_output)
        self.assertIn('modulated_features', emotional_output)
        self.assertEqual(emotional_output['emotions'].shape[1], 4)  # 4 emotions
        
    def test_decision_making(self):
        """Test decision making system"""
        # Create test inputs
        sensory = torch.randn(self.batch_size, self.brain.hidden_dim)
        memory = torch.randn(self.batch_size, self.brain.hidden_dim)
        emotional = torch.randn(self.batch_size, self.brain.hidden_dim)
        
        # Process through decision system
        decision_output = self.brain.make_decision(sensory, memory, emotional)
        
        # Verify outputs
        self.assertIn('action', decision_output)
        self.assertIn('response', decision_output)
        
    def test_learning_system(self):
        """Test learning system"""
        # Create test inputs
        features = torch.randn(self.batch_size, self.brain.hidden_dim * 2)
        reward = torch.tensor([[1.0]])
        
        # Process through learning system
        learning_output = self.brain.learn(features, reward)
        
        # Verify outputs
        self.assertIn('prediction_error', learning_output)
        self.assertIn('adapted_features', learning_output)
        
    def test_full_integration(self):
        """Test full integration of all systems"""
        # Create test inputs
        visual_input = torch.randn(self.batch_size, self.input_dim)
        auditory_input = torch.randn(self.batch_size, self.input_dim)
        current_emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        
        # Process through entire brain
        output = self.brain(visual_input, auditory_input, current_emotions)
        
        # Verify all components are present
        self.assertIn('sensory', output)
        self.assertIn('memory', output)
        self.assertIn('emotional', output)
        self.assertIn('decision', output)
        self.assertIn('integrated', output)
        
        # Verify emotional continuity
        self.assertTrue(torch.all(output['emotional']['emotions'] >= 0))
        self.assertTrue(torch.all(output['emotional']['emotions'] <= 1))
        
    def test_memory_persistence(self):
        """Test memory persistence across multiple steps"""
        # Create sequence of inputs
        sequence_length = 5
        visual_inputs = torch.randn(sequence_length, self.batch_size, self.input_dim)
        auditory_inputs = torch.randn(sequence_length, self.batch_size, self.input_dim)
        emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        
        # Process sequence
        memory_outputs = []
        for i in range(sequence_length):
            output = self.brain(visual_inputs[i], auditory_inputs[i], emotions)
            memory_outputs.append(output['memory']['integrated_memory'])
            
        # Verify memory persistence
        memory_tensor = torch.stack(memory_outputs)
        memory_diff = torch.diff(memory_tensor, dim=0)
        
        # Memory should change smoothly (not erratically)
        self.assertTrue(torch.all(torch.abs(memory_diff) < 0.5))
        
    def test_emotional_regulation(self):
        """Test emotional regulation capabilities"""
        # Create extreme input
        visual_input = torch.randn(self.batch_size, self.input_dim) * 2
        auditory_input = torch.randn(self.batch_size, self.input_dim) * 2
        high_fear = torch.tensor([[0.2, 0.2, 0.9, 0.7]])  # High fear and surprise
        
        # Process through brain
        output = self.brain(visual_input, auditory_input, high_fear)
        
        # Get new emotions
        new_emotions = output['emotional']['emotions']
        
        # Verify emotional regulation (emotions shouldn't spike to extremes)
        self.assertTrue(torch.all(new_emotions <= 0.95))  # Allow some high values but not 1.0
        self.assertTrue(torch.all(new_emotions >= 0.05))  # Allow some low values but not 0.0
        
    def test_developmental_integration(self):
        """Test integration with developmental stages"""
        # Create child at different stages
        stages = [
            DevelopmentalStage.NEWBORN,
            DevelopmentalStage.INFANT,
            DevelopmentalStage.EARLY_TODDLER
        ]
        
        for stage in stages:
            child = DigitalChild(initial_stage=stage)
            
            # Create test input
            visual_input = torch.randn(self.batch_size, self.input_dim)
            auditory_input = torch.randn(self.batch_size, self.input_dim)
            emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
            
            # Process through brain
            output = child.brain(visual_input, auditory_input, emotions)
            
            # Print debug info
            print(f"\nStage: {stage.name}")
            print(f"Integration level: {torch.mean(output['integrated']).item():.3f}")

            # Verify stage-appropriate processing
            if stage == DevelopmentalStage.NEWBORN:
                # Newborns should have simpler processing
                self.assertTrue(torch.mean(output['integrated']) < 0.5)
            elif stage == DevelopmentalStage.INFANT:
                # Infants should show moderate integration
                mean_integration = torch.mean(output['integrated']).item()
                self.assertTrue(0.3 < mean_integration < 0.7,
                              f"Integration level {mean_integration:.3f} not in range (0.3, 0.7)")
            else:  # EARLY_TODDLER
                # Toddlers should show more complex integration
                self.assertTrue(torch.mean(output['integrated']) > 0.4)

if __name__ == '__main__':
    unittest.main() 