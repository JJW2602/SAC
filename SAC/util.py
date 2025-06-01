import torch
import numpy as np
import gymnasium as gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_mlp():
    pass

def from_numpy(data):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    else:
        data = torch.from_numpy(data)
        if data.dtype == torch.float64:
            data = data.float()
        return data.to(device)

def sample_trajectory(
    env: gym.Env, policy , max_length: int, render: bool = False
    ):

    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        action = policy.get_action(ob)
        next_ob, reward, done, truncated, info = env.step(action)
        steps += 1
        rollout_done = done or steps > max_length 

        obs.append(ob)
        acs.append(action)
        rewards.append(reward)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob 

        if rollout_done:
            break

    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    env.close()

    return {
        "observation": np.array(obs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "episode_statistics": episode_statistics,
    }

def sample_n_trajectories(
    env: gym.Env, policy , ntraj: int, max_length: int, render: bool = False
):
    trajs = []
    for _ in range(ntraj):
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs