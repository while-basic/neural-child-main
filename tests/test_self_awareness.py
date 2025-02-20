# tests/test_self_awareness.py
# Description: Tests for self-awareness network functionality
# Created by: Christopher Celaya

import pytest
import torch
from main import DigitalChild, DevelopmentalStage
from self_awareness_network import SelfAwarenessNetwork, SelfAwarenessLevel

@pytest.fixture
def digital_child():
    return DigitalChild(initial_stage=DevelopmentalStage.EARLY_TODDLER)

@pytest.fixture
def self_awareness_network():
    return SelfAwarenessNetwork(input_dim=384, hidden_dim=256)

def test_self_awareness_initialization(digital_child):
    """Test that self-awareness network is properly initialized"""
    assert digital_child.self_awareness is not None
    assert isinstance(digital_child.self_awareness, SelfAwarenessNetwork)
    assert digital_child.self_awareness.current_level == SelfAwarenessLevel.PHYSICAL

def test_self_awareness_stage_progression(digital_child):
    """Test that self-awareness level updates with developmental stage"""
    # Start at EARLY_TODDLER (should be PHYSICAL awareness)
    assert digital_child.self_awareness.current_level == SelfAwarenessLevel.PHYSICAL
    
    # Progress to LATE_TODDLER (should develop MIRROR awareness)
    digital_child.set_stage(DevelopmentalStage.LATE_TODDLER, 0.5)
    assert digital_child.self_awareness.current_level == SelfAwarenessLevel.MIRROR
    
    # Progress to EARLY_PRESCHOOL (should develop EMOTIONAL awareness)
    digital_child.set_stage(DevelopmentalStage.EARLY_PRESCHOOL, 0.7)
    assert digital_child.self_awareness.current_level == SelfAwarenessLevel.EMOTIONAL

def test_self_awareness_processing(self_awareness_network):
    """Test self-awareness network processing"""
    batch_size = 1
    input_dim = 384
    hidden_dim = 256
    
    # Create test inputs
    sensory_input = torch.randn(batch_size, input_dim)
    emotional_state = torch.tensor([[0.7, 0.6, 0.2, 0.3]])  # [joy, trust, fear, surprise]
    memory_context = torch.randn(batch_size, input_dim)
    
    # Process through network
    outputs = self_awareness_network(
        sensory_input=sensory_input,
        emotional_state=emotional_state,
        memory_context=memory_context
    )
    
    # Verify outputs
    assert 'physical_features' in outputs
    assert 'emotional_features' in outputs
    assert 'cognitive_features' in outputs
    assert 'combined_features' in outputs
    assert 'attention_weights' in outputs
    assert 'reflection' in outputs
    
    # Check shapes
    assert outputs['physical_features'].shape == (batch_size, hidden_dim // 2)
    assert outputs['emotional_features'].shape == (batch_size, hidden_dim // 2)
    assert outputs['cognitive_features'].shape == (batch_size, hidden_dim // 2)
    assert outputs['combined_features'].shape == (batch_size, hidden_dim * 3 // 2)
    assert outputs['reflection'].shape == (batch_size, hidden_dim)

def test_self_concept_updating(self_awareness_network):
    """Test updating self-concept graph"""
    # Add some test experiences
    experiences = [
        {'content': 'I like playing with blocks'},
        {'content': 'I feel happy when I see mom'},
        {'content': 'I can stack blocks high'}
    ]
    
    for exp in experiences:
        self_awareness_network.update_self_concept(exp)
    
    # Check graph structure
    assert len(self_awareness_network.self_concept_graph.nodes) == 3
    assert len(self_awareness_network.self_concept_graph.edges) > 0  # Should have some connections

def test_development_metrics(self_awareness_network):
    """Test getting development metrics"""
    # Process some test data to generate metrics
    batch_size = 1
    sensory_input = torch.randn(batch_size, 384)
    emotional_state = torch.tensor([[0.7, 0.6, 0.2, 0.3]])
    memory_context = torch.randn(batch_size, 384)
    
    # Run a few forward passes
    for _ in range(5):
        self_awareness_network(
            sensory_input=sensory_input,
            emotional_state=emotional_state,
            memory_context=memory_context
        )
    
    # Get metrics
    metrics = self_awareness_network.get_development_metrics()
    
    # Verify metrics structure
    assert 'current_level' in metrics
    assert 'level_progress' in metrics
    assert 'metacognition' in metrics
    assert 'recent_reflections' in metrics
    assert len(metrics['recent_reflections']) == 5  # Should have 5 reflections

def test_state_saving_loading(tmp_path, self_awareness_network):
    """Test saving and loading network state"""
    # Process some test data
    batch_size = 1
    sensory_input = torch.randn(batch_size, 384)
    emotional_state = torch.tensor([[0.7, 0.6, 0.2, 0.3]])
    memory_context = torch.randn(batch_size, 384)
    
    # Run forward pass and update self-concept
    outputs = self_awareness_network(
        sensory_input=sensory_input,
        emotional_state=emotional_state,
        memory_context=memory_context
    )
    
    self_awareness_network.update_self_concept({
        'content': 'Test experience'
    })
    
    # Save state
    save_path = tmp_path / "self_awareness_test.pt"
    self_awareness_network.save_state(str(save_path))
    
    # Create new network and load state
    new_network = SelfAwarenessNetwork(input_dim=384, hidden_dim=256)
    new_network.load_state(str(save_path))
    
    # Verify state loaded correctly
    assert new_network.current_level == self_awareness_network.current_level
    assert new_network.level_progress == self_awareness_network.level_progress
    assert len(new_network.reflection_history) == len(self_awareness_network.reflection_history)
    assert len(new_network.self_concept_graph.nodes) == len(self_awareness_network.self_concept_graph.nodes) 