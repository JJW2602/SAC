import torch
import config as cfig
import numpy as np
import utils.utils as ut
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D

class ValueFunction(nn.Module):
    #nn.seq->reusable
    def __init__(self, obs_dim, out_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(obs_dim, cfig.hidden_size)
        self.fc2 = nn.Linear(cfig.hidden_size, cfig.hidden_size)
        self.fc3 = nn.Linear(cfig.hidden_size, out_dim)
        self.critic_state_optimizer = torch.optim.Adam(self.parameters(), lr=cfig.learning_rate)

    def forward(self, x):
        if x.dim() == 1:
           x = x.unsqueeze(0)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
class QFunction(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, cfig.hidden_size)
        self.fc2 = nn.Linear(cfig.hidden_size, cfig.hidden_size)
        self.fc3 = nn.Linear(cfig.hidden_size, out_dim)
        self.critic_optimizer = torch.optim.Adam(self.parameters(), lr=cfig.learning_rate)

    def forward(self, obs, actions):
        obs = self.flatten(obs)
        actions = self.flatten(actions)
        x = torch.cat([obs,actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
        

class Policy(nn.Module): 
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(obs_dim, cfig.hidden_size)
        self.fc2 = nn.Linear(cfig.hidden_size, cfig.hidden_size)
        self.mean = nn.Linear(cfig.hidden_size, action_dim)
        self.std = nn.Linear(cfig.hidden_size, action_dim)
        self.std_plus = nn.Softplus()
        self.policy_optimizer = torch.optim.Adam(self.parameters(), lr=cfig.learning_rate)

    def forward(self, obs : torch.tensor):
        if obs.dim() == 1:
           obs = obs.unsqueeze(0) 
        x=self.flatten(obs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std_plus(self.std(x))

        return  D.Independent(
            D.TransformedDistribution(
                base_distribution = D.Normal(mean, std),
                transforms = [D.TanhTransform(cache_size=1)]
            ),
            reinterpreted_batch_ndims=1
        )
     
    def get_action(self, obs: np.ndarray):
        obs = ut.from_numpy(obs)
        action_distribution = self(obs)
        sample_action = action_distribution.sample()

        return ut.to_numpy(sample_action).squeeze()