"""
Q-Learning Module for Neural Child Development
Created by: Christopher Celaya

This module implements Q-Learning for the neural child development system,
enabling reinforcement learning capabilities for decision making and behavior learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any, Union
import random

class QNetwork(nn.Module):
    """Q-Network for learning action-value functions"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 learning_rate: float = 0.001,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize Q-Network
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension
            learning_rate (float): Learning rate for optimizer
            memory_size (int): Size of replay memory
            batch_size (int): Size of training batch
            gamma (float): Discount factor
            epsilon_start (float): Starting exploration rate
            epsilon_end (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Neural network layers
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Target network for stable learning
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.layers.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.layers(state)
        
    def select_action(self, state: torch.Tensor) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state (torch.Tensor): Current state
            
        Returns:
            int: Selected action index
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.forward(state)
            return q_values.argmax().item()
            
    def store_transition(self, state: torch.Tensor, action: Union[int, torch.Tensor], 
                        reward: float, next_state: torch.Tensor):
        """Store transition in replay memory"""
        # Convert state and next_state to tensors if they aren't already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
        # Convert action to tensor if it's an integer
        if isinstance(action, int):
            action = torch.tensor([action], dtype=torch.long, device=self.device)
        elif isinstance(action, torch.Tensor):
            action = action.to(self.device)
            
        # Move tensors to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        
        # Store transition
        self.memory.append((state, action, reward, next_state))
        
    def update(self) -> float:
        """
        Update network parameters using replay memory
        
        Returns:
            float: Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).squeeze().to(self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        
        # Get current Q values
        current_q_values = self.forward(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - next_q_values.detach()) * self.gamma * next_q_values
            
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.layers.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
        
    def _update_target_network(self):
        """Update target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.layers.parameters()):
            target_param.data.copy_(param.data)
            
    def get_state_dict(self) -> Dict[str, Any]:
        """Get network state dictionary"""
        return {
            'network_state': self.layers.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load network state dictionary"""
        self.layers.load_state_dict(state_dict['network_state'])
        self.target_network.load_state_dict(state_dict['target_network_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.epsilon = state_dict['epsilon']

class QLearningSystem:
    """Q-Learning system for neural child development"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 device: str = "cpu"):
        """
        Initialize Q-Learning system
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension
            device (str): Device to use for computation
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize Q-Network
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.memory = []
        self.batch_size = 64
        self.learning_rate = 0.001
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        
    def select_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy"""
        state = state.to(self.device)
        return self.q_network.select_action(state)
        
    def store_transition(self, state: torch.Tensor, action: Union[int, torch.Tensor], 
                        reward: float, next_state: torch.Tensor):
        """Store transition in replay memory"""
        # Convert state and next_state to tensors if they aren't already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
        # Convert action to tensor if it's an integer
        if isinstance(action, int):
            action = torch.tensor([action], dtype=torch.long, device=self.device)
        elif isinstance(action, torch.Tensor):
            action = action.to(self.device)
            
        # Move tensors to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        
        # Store transition
        self.memory.append((state, action, reward, next_state))
        
    def train_step(self) -> float:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample random batch from memory
        batch_indices = torch.randint(len(self.memory), (self.batch_size,))
        states = []
        actions = []
        rewards = []
        next_states = []
        
        for idx in batch_indices:
            state, action, reward, next_state = self.memory[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).squeeze().to(self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        
        # Update network
        loss = self.q_network.update()
        self.loss_history.append(loss)
        
        return loss
        
    def get_average_reward(self) -> float:
        """Get average reward over last 100 episodes"""
        if not self.episode_rewards:
            return 0.0
        return sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
        
    def get_average_loss(self) -> float:
        """Get average loss over last 100 training steps"""
        if not self.loss_history:
            return 0.0
        return sum(self.loss_history[-100:]) / min(len(self.loss_history), 100)
        
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary"""
        return {
            'q_network': self.q_network.get_state_dict(),
            'memory': self.memory,
            'metrics': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'loss_history': self.loss_history
            }
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary"""
        self.q_network.load_state_dict(state_dict['q_network'])
        self.memory = state_dict['memory']
        self.episode_rewards = state_dict['metrics']['episode_rewards']
        self.episode_lengths = state_dict['metrics']['episode_lengths']
        self.loss_history = state_dict['metrics']['loss_history']
        
    def train_episode(self, 
                     get_state_fn,
                     get_reward_fn,
                     step_env_fn,
                     max_steps: int = 1000) -> Dict[str, float]:
        """
        Train for one episode
        
        Args:
            get_state_fn: Function to get current state
            get_reward_fn: Function to get reward
            step_env_fn: Function to step environment
            max_steps (int): Maximum steps per episode
            
        Returns:
            Dict[str, float]: Episode statistics
        """
        total_reward = 0
        steps = 0
        losses = []
        
        # Get initial state
        state = get_state_fn()
        
        for step in range(max_steps):
            # Select action
            action = self.q_network.select_action(state)
            
            # Take action
            next_state = step_env_fn(action)
            reward = get_reward_fn()
            done = step >= max_steps - 1
            
            # Store transition
            self.q_network.store_transition(state, action, reward, next_state)
            
            # Update network
            loss = self.q_network.update()
            if loss > 0:
                losses.append(loss)
                
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
                
        # Store episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        if losses:
            avg_loss = sum(losses) / len(losses)
            self.loss_history.append(avg_loss)
            
        return {
            'total_reward': total_reward,
            'steps': steps,
            'average_loss': avg_loss if losses else 0.0,
            'final_epsilon': self.q_network.epsilon
        }
        
    def get_action(self, state: torch.Tensor) -> int:
        """Get action for given state"""
        return self.q_network.select_action(state)
        
    def save_model(self, path: str):
        """Save model state"""
        torch.save(self.q_network.get_state_dict(), path)
        
    def load_model(self, path: str):
        """Load model state"""
        state_dict = torch.load(path)
        self.q_network.load_state_dict(state_dict)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions in the given state.
        
        Args:
            state (torch.Tensor): Current state tensor
            
        Returns:
            torch.Tensor: Q-values for all possible actions
        """
        state = state.to(self.device)
        with torch.no_grad():
            return self.q_network(state) 