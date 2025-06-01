from agent import ValueFunction, QFunction, Policy
from config import *
from torch import nn
import torch
import torch.nn.functional as F


class Agent():
    def __init__(self, obs_dim, action_dim):
            self.q_function1 = QFunction(obs_dim + action_dim, 1)
            self.q_function2 = QFunction(obs_dim + action_dim, 1)
            self.value_function = ValueFunction(obs_dim, 1)
            self.target_value_function = ValueFunction(obs_dim, 1)
            self.policy = Policy(obs_dim, action_dim)
            self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
            
            self.gamma = discount

            self.q_function1.to(self.device)
            self.q_function2.to(self.device)
            self.value_function.to(self.device)
            self.target_value_function.to(self.device)
            self.policy.to(self.device)

    def v_update(self, batch):
        obs = batch['observations']
        done = batch['done']

        with torch.no_grad():
            action_distribution = self.policy.squashed_distribution(obs)
            sample_actions = self.policy.get_action(obs)
            q_min = torch.min(self.q_function1(obs, sample_actions), self.q_function2(obs,sample_actions))
            target_value = torch.mean((q_min - self.entropy(action_distribution)), dim = 0)

        loss = F.mse_loss(self.value_function(obs), target_value) 

        self.value_function.critic_state_optimizer.zero_grad()
        loss.backward()
        self.value_function.critic_state_optimizer.step()


    def soft_target_update(self):  
        for target_param, param in zip(
            self.target_value_function.parameters(), self.value_function.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        

    def entropy(self, action_distribution):
        action = action_distribution.rsample()
        entropy = -action_distribution.log_prob(action)

        return entropy

    def q_update(self, batch):
        obs = batch['observations']
        rewards = batch['rewards']
        actions = batch['actions']
        next_obs = batch['next_observations']
        with torch.no_grad():
            target = rewards + self.gamma * self.target_value_function(next_obs)
        loss1 = F.mse_loss(self.q_function1(obs,actions), target)
        loss2 = F.mse_loss(self.q_function2(obs,actions), target)

        self.q_function1.critic_optimizer.zero_grad()
        loss1.backward()
        self.q_function1.critic_optimizer.step()

        self.q_function2.critic_optimizer.zero_grad()
        loss2.backward()
        self.q_function2.critic_optimizer.step()
    
    def policy_update(self, batch):
        obs = batch['observations']

        action_distribution = self.policy.squashed_distribution(obs)
        action = action_distribution.rsample()
        with torch.no_grad():
            min_q = min(self.q_function1(obs,action), self.q_function2(obs,action))
        
        loss = -torch.mean(min_q) -temperature * torch.mean(self.entropy(action_distribution))

        self.policy.policy_optimizer.zero_grad()
        loss.backward()
        self.policy.policy_optimizer.step()

    def update(self,batch):
        self.v_update(batch)
        self.soft_target_update()
        self.q_update(batch)
        self.policy_update(batch)