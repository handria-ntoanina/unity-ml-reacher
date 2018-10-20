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
        hidden_layers_size = [state_size, 600, 400, action_size]
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1]) for i in range(len(hidden_layers_size) - 1)])
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.hidden_layers)-1):
            layer = self.hidden_layers[i]
            layer.weight.data.uniform_(*hidden_init(layer))
        last = self.hidden_layers[-1]
        last.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Build a network that maps state to actions."""
        x = state
        for i in range(len(self.hidden_layers)):
            linear = self.hidden_layers[i]
            x = F.tanh(linear(x))
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
        self.insert_action_at = 2
        hidden_layers_size = [state_size, 600, 400, 200, 1]
        hidden_layers_size = [[hidden_layers_size[i], hidden_layers_size[i + 1]] for i in range(len(hidden_layers_size) - 1)]
        hidden_layers_size[self.insert_action_at][0] += action_size
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_layers_size[i][0], hidden_layers_size[i][1]) for i in range(len(hidden_layers_size))])
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.hidden_layers)-1):
            layer = self.hidden_layers[i]
            layer.weight.data.uniform_(*hidden_init(layer))
        last = self.hidden_layers[-1]
        last.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a network that maps state to actions."""
        x = state
        for i in range(len(self.hidden_layers)):
            if i == self.insert_action_at:
                x = torch.cat((x, action), dim=1)
            linear = self.hidden_layers[i]
            x = F.tanh(linear(x))
        return x