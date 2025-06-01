from gymnasium.wrappers import RescaleAction, ClipAction, RecordEpisodeStatistics
import gymnasium as gym

env_name : str = 'Hopper-v4'
training_starts : int = 10000
total_steps :int = 300000
sample_batch_size :int  = 128
max_capacity : int = 1000000
discount : float = 0.99
temperature : float = 0.1
num_layers : int = 3
eval_period : int = 1000
hidden_size : int = 128
tau : float = 0.005
learning_rate : float = 3e-4
num_eval_trajectories : int = 10

def make_env(render: bool = False):
    return RecordEpisodeStatistics(
        ClipAction(
            RescaleAction(
                gym.make(
                    env_name, render_mode="single_rgb_array" if render else None
                ),
                -1,
                1,
            )
        )
    )