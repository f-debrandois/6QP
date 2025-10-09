"""
Neural Network Models for RL Agents
Policy and Value networks for PPO and REINFORCE agents
"""

import torch
import torch.nn as nn
from game import NB_CARDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PolicyNetwork(nn.Module):
    """Neural network for RL agent policy"""
    def __init__(self, state_dim=116, hidden_dim=512):
        super(PolicyNetwork, self).__init__()
        # State: 4 row last values + 4 row lengths + 4 row penalties + 104 hand mask = 116
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, NB_CARDS)  # Output for each possible card
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x  # Return logits (will apply softmax with masking later)


class ValueNetwork(nn.Module):
    """Neural network for state value estimation (PPO critic)"""
    def __init__(self, state_dim=116, hidden_dim=512):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)  # Output single value
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)  # Return scalar value


class PPOMemory:
    """Experience buffer for PPO training"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_batches(self):
        """Return all experiences as tensors"""
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        values = torch.stack(self.values)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        
        return states, actions, old_log_probs, rewards, values, dones
