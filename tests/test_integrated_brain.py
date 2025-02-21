# tests/test_integrated_brain.py
# Description: Tests for the integrated brain architecture
# Created by: Christopher Celaya

import unittest
import torch
import numpy as np
from main import IntegratedBrain, DigitalChild
from developmental_stages import DevelopmentalStage

class TestIntegratedBrain(unittest.TestCase):
    def setUp(self):
        self.brain = IntegratedBrain()
        self.device = self.brain.device  # Use the brain's device
        self.input_dim = 384
        self.batch_size = 1
        
    def test_sensory_processing(self):
        """Test sensory system processing"""
        # Create test inputs
        visual_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
        auditory_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
        
        # Process through sensory system
        sensory_output = self.brain.process_sensory_input(visual_input, auditory_input)
        
        # Verify output
        self.assertIn('sensory_features', sensory_output)
        self.assertEqual(sensory_output['sensory_features'].shape[0], self.batch_size)
        
    def test_memory_processing(self):
        """Test memory system processing"""
        # Create test input
        features = torch.randn(self.batch_size, self.brain.hidden_dim).to(self.device)
        
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
        features = torch.randn(self.batch_size, self.brain.hidden_dim).to(self.device)
        current_emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=self.device)  # [joy, trust, fear, surprise]
        
        # Process through emotional system
        emotional_output = self.brain.process_emotions(features, current_emotions)
        
        # Verify outputs
        self.assertIn('emotions', emotional_output)
        self.assertIn('modulated_features', emotional_output)
        self.assertEqual(emotional_output['emotions'].shape[1], 4)  # 4 emotions
        
    def test_decision_making(self):
        """Test decision making system"""
        # Create test inputs
        features = torch.randn(self.batch_size, self.brain.hidden_dim * 3).to(self.device)  # Combined features
        
        # Process through decision system
        decision_output = self.brain.make_decision(features)
        
        # Verify outputs
        self.assertIn('action', decision_output)
        self.assertIn('response', decision_output)
        
    def test_learning_system(self):
        """Test learning system"""
        # Create test inputs
        features = torch.randn(self.batch_size, self.brain.hidden_dim * 2).to(self.device)
        reward = torch.tensor([[1.0]], device=self.device)
        
        # Process through learning system
        learning_output = self.brain.learn(features, reward)
        
        # Verify outputs
        self.assertIn('prediction_error', learning_output)
        self.assertIn('adapted_features', learning_output)
        
    def test_full_integration(self):
        """Test full integration of all systems"""
        # Create test inputs
        visual_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
        auditory_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
        current_emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=self.device)
        
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
        visual_inputs = torch.randn(sequence_length, self.batch_size, self.input_dim).to(self.device)
        auditory_inputs = torch.randn(sequence_length, self.batch_size, self.input_dim).to(self.device)
        emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=self.device)
        
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
        visual_input = torch.randn(self.batch_size, self.input_dim).to(self.device) * 2
        auditory_input = torch.randn(self.batch_size, self.input_dim).to(self.device) * 2
        high_fear = torch.tensor([[0.2, 0.2, 0.9, 0.7]], device=self.device)  # High fear and surprise
        
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
            child = DigitalChild(stage=stage)  # Changed from initial_stage to stage
            
            # Create test input
            visual_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
            auditory_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
            emotions = torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=self.device)
            
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

    def test_enhanced_reward_system(self):
        """Test the enhanced reward system with all components."""
        # Test case 1: Positive interaction with good alignment
        user_input_positive = "I'm really excited about learning new things!"
        ai_response_positive = "That's wonderful! I'm curious to learn more about what interests you. What specific topics would you like to explore?"
        
        reward_positive = self.brain.get_reward(user_input_positive, ai_response_positive)
        self.assertGreater(reward_positive, 0.5, "Positive interaction should yield high reward")
        
        # Test case 2: Emotional misalignment
        user_input_sad = "I'm feeling quite sad today."
        ai_response_happy = "That's awesome! Let's celebrate!"
        
        reward_misaligned = self.brain.get_reward(user_input_sad, ai_response_happy)
        self.assertLess(reward_misaligned, 0.0, "Emotional misalignment should yield negative reward")
        
        # Test case 3: Good memory recall
        # First store something in memory
        memory_text = "My favorite color is blue"
        memory_features = torch.randn(1, self.brain.hidden_dim // 4).to(self.device)
        self.brain.memory_states['working'] = memory_features.squeeze()
        
        user_input_memory = "What's my favorite color?"
        ai_response_memory = "Based on our previous conversation, your favorite color is blue!"
        
        reward_memory = self.brain.get_reward(user_input_memory, ai_response_memory)
        self.assertGreater(reward_memory, 0.3, "Good memory recall should yield positive reward")
        
        # Test case 4: Curiosity and exploration
        user_input_basic = "Tell me about space."
        ai_response_curious = "Space is fascinating! I'm curious about what aspects interest you most. Shall we explore the mysteries of black holes, or would you like to learn about distant galaxies?"
        
        reward_curious = self.brain.get_reward(user_input_basic, ai_response_curious)
        self.assertGreater(reward_curious, 0.6, "Curious response should yield high reward")
        
        # Test case 5: Poor conversation flow (repetitive)
        user_input_repeat = "How are you?"
        ai_response_repeat = "I'm doing well"  # Add this to short-term memory
        self.brain.memory_states['short_term'] = torch.tensor([ord(c) for c in ai_response_repeat], device=self.device)
        
        reward_repeat = self.brain.get_reward(user_input_repeat, ai_response_repeat)
        self.assertLess(reward_repeat, 0.5, "Repetitive response should yield lower reward")
        
        # Test case 6: No input/response (base reward only)
        base_reward = self.brain.get_reward()
        self.assertIsInstance(base_reward, float, "Base reward should be a float")
        self.assertGreaterEqual(base_reward, -1.0, "Reward should be >= -1.0")
        self.assertLessEqual(base_reward, 1.0, "Reward should be <= 1.0")
        
        # Print reward metrics for analysis
        print("\nReward System Test Results:")
        print(f"Positive interaction reward: {reward_positive:.3f}")
        print(f"Emotional misalignment reward: {reward_misaligned:.3f}")
        print(f"Memory recall reward: {reward_memory:.3f}")
        print(f"Curiosity reward: {reward_curious:.3f}")
        print(f"Repetitive response reward: {reward_repeat:.3f}")
        print(f"Base reward: {base_reward:.3f}")

if __name__ == '__main__':
    unittest.main() 