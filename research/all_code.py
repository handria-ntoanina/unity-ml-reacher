from collections import deque
import torch
import torch.tensor as tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

class Config:
    DEVICE = torch.device('cpu')
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.discount = None
        self.gradient_clip = None
        self.entropy_weight = 0.01
        self.gae_tau = 1.0
        self.rollout_length = None
        self.optimization_epochs = 4
        self.num_mini_batches = 32
        self.state_dim = None
        self.action_dim = None

class BaseAgent:
    def __init__(self, config):
        self.config = config

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        
class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.scores_list = []
        self.scores_deque = deque(maxlen=100)
        self.score_max = 0

    def step(self):
        config = self.config
        rollout = []
        states = self.task.reset()
        online_rewards = np.zeros(len(states))
        for _ in range(config.rollout_length):
            actions, log_probs, _, values = self.network(states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            online_rewards += rewards
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states
            
        self.score_max = np.max(online_rewards)
        self.scores_deque.append(np.mean(online_rewards))
        self.scores_list.append(np.mean(online_rewards))
                
        pending_value = self.network(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((len(states), 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
            advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            for indices in batch_indices(len(states), config.num_mini_batches):
                indices = tensor(indices).long()
                sampled_states = states[indices]
                sampled_actions = actions[indices]
                sampled_log_probs_old = log_probs_old[indices]
                sampled_returns = returns[indices]
                sampled_advantages = advantages[indices]

                _, log_probs, entropy_loss, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(128, 128), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x
        
class GaussianActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi_a = self.actor_body(obs)
        phi_v = self.critic_body(obs)
        mean = F.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v
    
def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def random_seed():
    np.random.seed()
    torch.manual_seed(np.random.randint(int(1e6)))
    
def batch_indices(length, batch_size):
    indices = np.arange(length)
    np.random.shuffle(indices)
    for i in range(1 + length // batch_size):
        start = batch_size*i
        end = start + batch_size
        end = min(length, end)
        if start >= length:
            return
        yield indices[start:end]