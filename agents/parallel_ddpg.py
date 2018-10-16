import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from agents.utils import soft_update, apply_noise, build_noise, timefunc

device = "cpu"
class ParallelDDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_agents, memory, ActorNetwork, CriticNetwork, device,
                GAMMA = 0.99, 
                TAU = 1e-3, 
                LR_CRITIC = 5e-4,
                LR_ACTOR = 5e-4, 
                UPDATE_EVERY = 1,
                TRANSFER_EVERY = 1,
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
            GAMMA: discount factor
            TAU: for soft update of target parameters
            LR_CRITIC: learning rate of the critics
            LR_ACTOR: learning rate of the actors
            UPDATE_EVERY: how often to update the networks
            TRANSFER_EVERY: how often to transfer from the online network to the targeted fixed network
            FILE_NAME: default prefix to the saved model
        """
        self.seed = random.seed(seed)
        self.num_agents = num_agents
        
        # Actor networks
        self.actors_local = [ActorNetwork(state_size, action_size, seed).to(device) for i in range(num_agents)]
        self.actors_optim = [optim.Adam(self.actors_local[i].parameters(), lr=LR_ACTOR) for i in range(num_agents)]
        self.actor_target = ActorNetwork(state_size, action_size, seed).to(device)
                 
        # Critic networks
        self.critics_local = [CriticNetwork(state_size, action_size, seed).to(device) for i in range(num_agents)]
        self.critics_optim = [optim.Adam(self.critics_local[i].parameters(), lr=LR_CRITIC) for i in range(num_agents)]
        self.critic_target = CriticNetwork(state_size, action_size, seed).to(device)
        
        self.device = device
        # Noise
        
        self.noise = build_noise(self.actor_target, device, seed)
        
        # Replay memory
        self.memory = memory
        # Initialize time steps (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.t_step = 0
        
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR_CRITIC = LR_CRITIC
        self.LR_ACTOR = LR_ACTOR
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TRANSFER_EVERY =TRANSFER_EVERY
     
    def save(self):
        for i in range(self.num_agents):
            torch.save(agent.critics_local[i].state_dict(),"{}_critic_{}.pth".format(self.FILE_NAME, i))
            torch.save(agent.actors_local[i].state_dict(),"{}_actor_{}.pth".format(self.FILE_NAME, i))
            torch.save(agent.critic_target.state_dict(),"{}_critic_{}.pth".format(self.FILE_NAME, "target"))
            torch.save(agent.actor_target.state_dict(),"{}_actor_{}.pth".format(self.FILE_NAME, "target"))
     
    def load(self):
        for i in range(self.num_agents):
            agent.critics_local[i].load_state_dict(torch.load("{}_critic_{}.pth".format(self.FILE_NAME, i)))
            agent.actors_local[i].load_state_dict(torch.load("{}_actor_{}.pth".format(self.FILE_NAME, i)))
            agent.critic_target.load_state_dict(torch.load("{}_critic_{}.pth".format(self.FILE_NAME, "target")))
            agent.actor_target.load_state_dict(torch.load("{}_actor_{}.pth".format(self.FILE_NAME, "target")))
    
    def reset(self):
        self.noise.reset()
    
    @timefunc
    def act(self, states, add_noise=True):
        """Returns actions of each actor for given states.
        
        Params
        ======
            state (array_like): current states
        """
        ret = None
        
        with torch.no_grad():
            if add_noise:
                noise_sample = self.noise.sample()
                for i in range(self.num_agents):
                    state = torch.from_numpy(states[i]).float().unsqueeze(0).to(device)
                    actor_local = self.actors_local[i]
                    actor_local.eval()
                    # apply a noise to the parameters only for exploration purpose
                    apply_noise(actor_local, noise_sample)
                    to_add = actor_local(state).cpu().data.numpy()
                    if ret is None:
                        ret = to_add
                    else:
                        ret = np.concatenate((ret, to_add))
                    # restore the previous parameters otherwise the noise will disturb the acquired knowldege
                    apply_noise(actor_local, -noise_sample)
                    actor_local.train()
            else:
                for i in range(self.num_agents):
                    state = torch.from_numpy(states[i]).float().unsqueeze(0).to(device)
                    actor_local = self.actors_local[i]
                    actor_local.eval()
                    if ret is None:
                        ret = to_add
                    else:
                        ret = np.concatenate((ret, to_add))
                    actor_local.train()
        return ret
    
    @timefunc
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.TRANSFER_EVERY
        self.u_step = (self.u_step + 1) % self.UPDATE_EVERY
        
        if len(self.memory) > self.memory.batch_size and self.u_step == 0:
            for i in range(self.num_agents):
                experiences = self.memory.sample()
                self.learn(experiences, self.actors_local[i], self.actors_optim[i], self.critics_local[i], self.critics_optim[i])
        
        if len(self.memory) > self.memory.batch_size and self.t_step == 0:
            for i in range(self.num_agents):
                soft_update(self.actors_local[i], self.actor_target, self.TAU)
                soft_update(self.critics_local[i], self.critic_target, self.TAU)
    
    @timefunc
    def learn(self, experiences, actor_local, actor_optim, critic_local, critic_optim):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences
        
        # The critic should estimate the value of the states to be equal to rewards plus
        # the estimation of the next_states value according to the critic_target and actor_target
        next_actions = self.actor_target(next_states)
        targeted_value = rewards + self.GAMMA*self.critic_target(next_states, next_actions)
        
        current_value = critic_local(states, actions)
        
        # calculate the loss and backprobagate
        critic_optim.zero_grad()
        F.mse_loss(current_value, targeted_value).backward()
        critic_optim.step()
        
        # the actor's objective is to increase/decrease its parameters according to the value returned by the critic_local
        # that is achieved by maximizing the return of the critic_local through a gradient ascent
        actions_pred = actor_local(states)
        actor_optim.zero_grad()
        (- critic_local(states, actions_pred).mean()).backward()
        actor_optim.step()

    