from agent.agent import ValueFunction, QFunction, Policy
import config as cfig
import utils.utils as ut
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

class Agent():
    def __init__(self, obs_dim, action_dim):
            self.q_function1 = QFunction(obs_dim + action_dim, 1).to(cfig.device)
            self.q_function2 = QFunction(obs_dim + action_dim, 1).to(cfig.device)
            self.value_function = ValueFunction(obs_dim, 1).to(cfig.device)
            self.target_value_function = ValueFunction(obs_dim, 1).to(cfig.device)
            self.target_value_function.load_state_dict(self.value_function.state_dict())
            self.policy = Policy(obs_dim, action_dim).to(cfig.device)
            self.gamma = cfig.discount

    def v_update(self, batch):
        obs : torch.tensor = batch['observations']
        
        with torch.no_grad():
            action_distribution = self.policy(obs)
            sample_actions = ut.from_numpy(self.policy.get_action(ut.to_numpy(obs)))
            #q_min이 제대로 출력되는지? -> self.q_function1, q_fuction2비교할 것. element-wise가 아닌 것을 확인할 것
            q_min = torch.min(self.q_function1(obs, sample_actions), self.q_function2(obs,sample_actions))
            target_value = q_min + self.entropy(action_distribution) #여기는 state마다 action을 하나씩 뽑아서 따로 mean을 안해도 되는건지??
 
        loss = F.mse_loss(self.value_function(obs), target_value) 

        self.value_function.critic_state_optimizer.zero_grad()
        loss.backward()
        self.value_function.critic_state_optimizer.step()


    def soft_target_update(self):  
        for target_param, param in zip(
            self.target_value_function.parameters(), self.value_function.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - cfig.tau) + param.data * cfig.tau
            )

    def entropy(self, action_distribution):
        action = action_distribution.rsample()
        entropy = -action_distribution.log_prob(action)
        return entropy


    def q_update(self, batch):
        obs = batch['observations']
        rewards = batch['rewards'].unsqueeze(-1)
        actions = batch['actions']
        next_obs = batch['next_observations']
        dones = batch['dones'].unsqueeze(-1)
         

        with torch.no_grad():
            target = rewards + (1.0 - dones.float()) * self.gamma * self.target_value_function(next_obs)
            #여기서 next_obs를 사용해도 되는게, dynamics는 변하지 않기 때문? -> value_function처럼 policy에서처럼 sampling x?
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
        action_distribution = self.policy(obs)
        actions = action_distribution.rsample()
        
        min_q = torch.min(self.q_function1(obs,actions), self.q_function2(obs,actions))
        loss = torch.mean(-cfig.temperature * self.entropy(action_distribution) - min_q)

        #backpropagation : 변수에 대해서 다 생성? step
        self.policy.policy_optimizer.zero_grad()
        loss.backward()
        self.policy.policy_optimizer.step()

    def update(self, batch):
        self.v_update(batch)
        self.soft_target_update()
        self.q_update(batch)
        self.policy_update(batch)