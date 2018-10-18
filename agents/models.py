import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

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
        hidden_layers_size = [state_size,  512, 256, action_size]
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1]) for i in range(len(hidden_layers_size) - 1)])
        self.normalization_layers = nn.ModuleList()
        self.normalization_layers.extend([nn.BatchNorm1d(hidden_layers_size[i]) for i in range(1,len(hidden_layers_size))])
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            layer.weight.data.uniform_(*hidden_init(layer))
#         last = self.hidden_layers[-1]
#         last.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Build a network that maps state to actions."""
        x = state
        for i in range(len(self.hidden_layers)-1):
            linear = self.hidden_layers[i]
            x = F.relu(linear(x))
            batch_norm = self.normalization_layers[i]
            x = batch_norm(x)
        last = self.hidden_layers[-1]
        x = torch.tanh(last(x))
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
        hidden_layers_size = [state_size + action_size, 512, 256, 128, 1]
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1]) for i in range(len(hidden_layers_size) - 1)])
        self.normalization_layers = nn.ModuleList()
        self.normalization_layers.extend([nn.BatchNorm1d(hidden_layers_size[i]) for i in range(1,len(hidden_layers_size))])
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            layer.weight.data.uniform_(*hidden_init(layer))
#         last = self.hidden_layers[-1]
#         last.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a network that maps state to actions."""
        x = torch.cat((state, action), dim=1)
        for i in range(len(self.hidden_layers)-1):
            linear = self.hidden_layers[i]
            x = F.relu(linear(x))
            x = self.normalization_layers[i](x)
        last = self.hidden_layers[-1]
        x = last(x)
        return x