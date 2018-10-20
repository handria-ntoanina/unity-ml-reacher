import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from agents.utils import soft_update
from agents.ddpg import DDPG


class MultiDDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed, num_agents, memory, ActorNetwork, CriticNetwork, device,
                BOOTSTRAP_SIZE = 5,
                GAMMA = 0.99, 
                TAU = 1e-3, 
                LR_CRITIC = 5e-4,
                LR_ACTOR = 5e-4, 
                UPDATE_EVERY = 1,
                TRANSFER_EVERY = 2,
                UPDATE_LOOP = 10,
                ADD_NOISE_EVERY = 5,
                WEIGHT_DECAY = 0,
                FILE_NAME="multi_ddpg"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            num_agents: number of running agents
            memory: instance of ReplayBuffer
            ActorNetwork: a class inheriting from torch.nn.Module that define the structure of the actor neural network
            CriticNetwork: a class inheriting from torch.nn.Module that define the structure of the critic neural network
            device: cpu or cuda:0 if available
            BOOTSTRAP_SIZE: length of the bootstrap
            GAMMA: discount factor
            TAU: for soft update of target parameters
            LR_CRITIC: learning rate of the critics
            LR_ACTOR: learning rate of the actors
            UPDATE_EVERY: how often to update the networks
            TRANSFER_EVERY: after how many update do we transfer from the online network to the targeted fixed network
            UPDATE_LOOP: Number of loops of learning whenever the agent is learning
            ADD_NOISE_EVERY: how often to add noise to favor exploration
            WEIGHT_DECAY: Parameter of the Adam Optimizer of the Critic Network
            FILE_NAME: default prefix to the saved model
        """
        # Instantiate n agent with n network
        self.agents = [DDPG(state_size, action_size, seed, memory, ActorNetwork, CriticNetwork, device, 
                BOOTSTRAP_SIZE ,
                GAMMA , 
                TAU , 
                LR_CRITIC ,
                LR_ACTOR , 
                UPDATE_EVERY ,
                TRANSFER_EVERY ,
                UPDATE_LOOP, 
                ADD_NOISE_EVERY ,
                WEIGHT_DECAY ,
                FILE_NAME=FILE_NAME + "_" + str(i)) for i in range(num_agents)]
        self.rewards = [deque(maxlen=BOOTSTRAP_SIZE) for i in range(num_agents)]
        self.states = [deque(maxlen=BOOTSTRAP_SIZE) for i in range(num_agents)]
        self.gammas = np.array([GAMMA ** i for i in range(BOOTSTRAP_SIZE)])
    
    def reset(self):
        for agent in self.agents:
            agent.noise.reset()
        
    def set_noise(self, noise):
        for agent in self.agents:
            agent.noise = noise
    
    def save(self):
        for agent in self.agents:
            agent.save()
     
    def load(self, folder="./"):
        for agent in self.agents:
            agent.load(folder)
    
    def act(self, states, add_noise=True):
        """Returns actions of each actor for given states.
        
        Params
        ======
            states (array_like): current states
        """
        return [self.agents[i].act(states[i], add_noise) for i in range(len(self.agents))]
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for i in range(len(self.agents)):
            self.agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i])