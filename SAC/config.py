from gymnasium.wrappers import RescaleAction, ClipAction, RecordEpisodeStatistics
import gymnasium as gym
import torch

env_name : str = 'Hopper-v4'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_steps :int = 300000
random_steps : int = 10000
training_starts : int = 10000

max_capacity : int = 1000000
sample_batch_size :int  = 256

discount : float = 0.99
temperature : float = 0.1
tau : float = 0.005
learning_rate : float = 3e-4
num_layers : int = 3
hidden_size : int = 256

eval_period : int = 1000
num_eval_trajectories : int = 10