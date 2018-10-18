import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from agents.utils import soft_update

device = "cpu"
class ParallelDDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_agents, memory, ActorNetwork, CriticNetwork, device,
                BOOTSTRAP_SIZE = 5,
                GAMMA = 0.99, 
                TAU = 1e-3, 
                LR_CRITIC = 5e-4,
                LR_ACTOR = 5e-4, 
                UPDATE_EVERY = 1,
                TRANSFER_EVERY = 4,
                ADD_NOISE_EVERY = 5,
                FILE_NAME="parallel_dpg"):
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
            BOOSTRAP_SIZE: length of the bootstrap
            GAMMA: discount factor
            TAU: for soft update of target parameters
            LR_CRITIC: learning rate of the critics
            LR_ACTOR: learning rate of the actors
            UPDATE_EVERY: how often to update the networks
            TRANSFER_EVERY: how often to transfer from the online network to the targeted fixed network
            ADD_NOISE_EVERY: how often to add noise to favor exploration
            FILE_NAME: default prefix to the saved model
        """
        self.seed = random.seed(seed)
        self.num_agents = num_agents
        
        # Actor networks
        self.actor_local = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.actor_target = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_target.eval()
                 
        # Critic networks
        self.critic_local = CriticNetwork(state_size, action_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        self.critic_target = CriticNetwork(state_size, action_size, seed).to(device)
        self.critic_target.eval()
        
        self.device = device
        
        # Noise
        self.noise = None
        
        # Replay memory
        self.memory = memory
        # Initialize time steps (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.t_step = 0
        self.n_step = 0
        
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR_CRITIC = LR_CRITIC
        self.LR_ACTOR = LR_ACTOR
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TRANSFER_EVERY = TRANSFER_EVERY
        self.ADD_NOISE_EVERY = ADD_NOISE_EVERY
        
        self.rewards = [deque(maxlen=BOOTSTRAP_SIZE) for i in range(self.num_agents)]
        self.states = [deque(maxlen=BOOTSTRAP_SIZE) for i in range(self.num_agents)]
        self.gammas = np.array([GAMMA ** i for i in range(BOOTSTRAP_SIZE)])
    
    def reset(self):
        if self.noise:
            self.noise.reset()
        
    def set_noise(self, noise):
        self.noise = noise
    
    def save(self):
        torch.save(self.critic_local.state_dict(),"{}_critic.pth".format(self.FILE_NAME))
        torch.save(self.actor_local.state_dict(),"{}_actor.pth".format(self.FILE_NAME))
        torch.save(self.critic_target.state_dict(),"{}_critic_{}.pth".format(self.FILE_NAME, "target"))
        torch.save(self.actor_target.state_dict(),"{}_actor_{}.pth".format(self.FILE_NAME, "target"))
     
    def load(self):
        self.critic_local.load_state_dict(torch.load("{}_critic.pth".format(self.FILE_NAME)))
        self.actor_local.load_state_dict(torch.load("{}_actor.pth".format(self.FILE_NAME)))
        self.critic_target.load_state_dict(torch.load("{}_critic_{}.pth".format(self.FILE_NAME, "target")))
        self.actor_target.load_state_dict(torch.load("{}_actor_{}.pth".format(self.FILE_NAME, "target")))
    
    def act(self, states, add_noise=True):
        """Returns actions of each actor for given states.
        
        Params
        ======
            state (array_like): current states
        """
        ret = None
        
        self.n_step = (self.n_step + 1) % self.ADD_NOISE_EVERY
        
        with torch.no_grad():
            if add_noise and self.noise and self.n_step == 0:
                self.actor_local.eval()
                for i in range(self.num_agents):
                    state = torch.from_numpy(states[i]).float().unsqueeze(0).to(device)
                    to_add = self.noise.apply(self.actor_local, state)
                    if ret is None:
                        ret = to_add
                    else:
                        ret = np.concatenate((ret, to_add))
                self.actor_local.train()
            else:
                self.actor_local.eval()
                for i in range(self.num_agents):
                    state = torch.from_numpy(states[i]).float().unsqueeze(0).to(device)
                    to_add = self.actor_local(state).cpu().data.numpy()
                    if ret is None:
                        ret = to_add
                    else:
                        ret = np.concatenate((ret, to_add))
                self.actor_local.train()
        return ret
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        
        for i in range(self.num_agents):
            self.rewards[i].append(rewards[i])
            self.states[i].append(states[i])
            if len(self.rewards[i])==5:
                reward = np.sum(self.rewards[i] * self.gammas)
                self.memory.add(self.states[i][0], actions[i], reward, next_states[i], np.any(dones))
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.TRANSFER_EVERY
        self.u_step = (self.u_step + 1) % self.UPDATE_EVERY
        
        if len(self.memory) > self.memory.batch_size and self.u_step == 0:
            self.learn()
        
        if len(self.memory) > self.memory.batch_size and self.t_step == 0:
            soft_update(self.actor_local, self.actor_target, self.TAU)
            soft_update(self.critic_local, self.critic_target, self.TAU)
    
    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        my_sum = 0.
        
        losses = torch.zeros(0)
        values = torch.zeros(0)
        for i in range(self.num_agents):
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences
        
            # The critic should estimate the value of the states to be equal to rewards plus
            # the estimation of the next_states value according to the critic_target and actor_target
            
            next_actions = self.actor_target(next_states)
            targeted_value = rewards + self.GAMMA*self.critic_target(next_states, next_actions)
            current_value = self.critic_local(states, actions)
        
            # calculate the loss
            loss = F.mse_loss(current_value, targeted_value)
            losses = torch.cat((losses, loss.unsqueeze(0)))
            
            # compile the values of the critic to optimize the actor 
            actions_pred = self.actor_local(states)
            mean = self.critic_local(states, actions_pred).mean()
            values = torch.cat((values, mean.unsqueeze(0)))
            
            
        self.critic_optim.zero_grad()
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()
       
        self.actor_optim.zero_grad()
        (-(values.mean())).backward()
        self.actor_optim.step()

    