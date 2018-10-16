import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of action space
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        hidden_layers_size = [state_size, 64, 32, action_size]
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1]) for i in range(len(hidden_layers_size) - 1)])
        self.normalization_layers = nn.ModuleList()
        self.normalization_layers.extend([nn.BatchNorm1d(hidden_layers_size[i]) for i in range(1,len(hidden_layers_size))])
    def forward(self, state):
        """Build a network that maps state to actions."""
        x = state
        for i in range(len(self.hidden_layers)-1):
            linear = self.hidden_layers[i]
            batch_norm = self.normalization_layers[i]
            x = batch_norm(F.relu(linear(x)))
        last = self.hidden_layers[-1]
        x = F.tanh(last(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of action space
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        hidden_layers_size = [state_size + action_size, 64, 32, 1]
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1]) for i in range(len(hidden_layers_size) - 1)])
        self.normalization_layers = nn.ModuleList()
        self.normalization_layers.extend([nn.BatchNorm1d(hidden_layers_size[i]) for i in range(1,len(hidden_layers_size))])
    def forward(self, state, action):
        """Build a network that maps state to actions."""
        x = torch.cat((state, action), dim=1)
        for i in range(len(self.hidden_layers)-1):
            linear = self.hidden_layers[i]
            batch_norm = self.normalization_layers[i]
            x = batch_norm(F.relu(linear(x)))
        last = self.hidden_layers[-1]
        x = last(x)
        return x