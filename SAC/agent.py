import torch
from config import *
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D

class ValueFunction(nn.Module):
    #nn.seq->reusable
    def __init__(self, obs_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_dim)
        self.critic_state_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.tau = tau

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
        
    

class QFunction(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_dim)
        self.gamma = discount
        self.critic_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs, actions):
        x = torch.cat([obs,actions], dim=1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
        

class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.std = nn.Linear(hidden_size, action_dim)
        self.temperature = temperature
        self.policy_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self,x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)

        return mean, std
    
    def squashed_distribution(self, obs):
        mean, std = self(obs)
        return D.Independent(
            D.TransformedDistribution(
                base_distribution = D.Normal(mean, std),
                transforms = [D.TanhTransform(cache_size=1)]
            ),
            reinterpreted_batch_ndims=1
        )

    def get_action(self, obs):
        mean, std = self(obs)
        sample_action = self.squashed_distribution(obs, mean, std).sample()
        return sample_action

        
        