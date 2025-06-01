import torch
import tqdm
import argparse
import wandb
import gymnasium as gym

from util import *
from config import *
from ReplayBuffer import ReplayBuffer
from agent import ValueFunction, QFunction, Policy
from SAC_agent import Agent
    
def running_training_loop():
    #1. Set Environment & device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_env()
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    #2. initialize instances(RB, Critic, Actor)
    replaybuffer = ReplayBuffer(max_capacity)

    agent = Agent(obs_dim, action_dim)

    #3. Interact wiht Env -> update(Actor & Critic)
    obs , _ = env.reset()
    for step in tqdm.trange(total_steps, dynamic_ncols=True):
        
        action = agent.policy.get_action(obs)

        next_obs, reward, done, trunc, _ = env.step(action)

        replaybuffer.store(obs, reward, action, next_obs, trunc, done)

        if done or trunc: #episode length
            obs, _ = env.reset()
        else : obs = next_obs

        #update
        if step >= training_starts :
            batch = replaybuffer.sample(sample_batch_size)
            batch = from_numpy(batch)
            
            agent.update()
            

        #evaluation
        if step % eval_period ==0:
            trajectories = sample_n_trajectories(
                env_name,
                policy=agent,
                ntraj=num_eval_trajectories,
                max_length=env.spec.max_episode_steps
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            avg_return = np.mean(returns)
            print(f"Returns at step {step} : {avg_return}")

            wandb.log({
                "Eval_AverageReturn": avg_return,
                "Step": step
            }, step=step)      
        
def main():
    wandb.init(
        project="SAC-Evaluation",  # 원하는 프로젝트 이름
        config={
            "total_steps": total_steps,
            "eval_period": eval_period,
            "sample_batch_size": sample_batch_size,
            "env_name": env_name,
        }
    )
    running_training_loop()


if __name__ == '__main__' :
    main()
    