import torch
import tqdm
import argparse
import wandb
import gymnasium as gym

import numpy as np
import utils.utils as ut
import config as cfig
from ReplayBuffer import ReplayBuffer
from agent.SAC_agent import Agent
    
def running_training_loop():
    #1. Set Environment & device
    env = gym.make(cfig.env_name)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    #2. initialize instances(RB, Critic, Actor)
    replaybuffer = ReplayBuffer(cfig.max_capacity)
    agent = Agent(obs_dim, action_dim)

    #3. Interact wiht Env -> update(Actor & Critic)
    obs , _ = env.reset()
    for step in tqdm.trange(cfig.total_steps, dynamic_ncols=True):
        if step < cfig.random_steps:
            action = env.action_space.sample()
        else:
            action : np.ndarray = agent.policy.get_action(obs)
        
        next_obs, reward, done, trunc, _ = env.step(action) #gym에서는 step을 batch단위로 x batch 1은 unsqueeze해야함
        replaybuffer.store(obs, action, reward, next_obs, done)

        if done or trunc: #episode finish
            obs, _ = env.reset()
        else : obs = next_obs

        #update
        if step >= cfig.training_starts :
            batch = replaybuffer.sample(cfig.sample_batch_size)
            batch = ut.from_numpy(batch)
            agent.update(batch)
            

        #evaluation
        '''
        if step % cfig.eval_period ==0:
            trajectories = sample_n_trajectories(
                env,
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
            '''

def main():
    
    '''wandb.init(
        project="SAC-Evaluation", 
        config={
            "total_steps": total_steps,
            "eval_period": eval_period,
            "sample_batch_size": sample_batch_size,
            "env_name": env_name,
        }
    )'''

    running_training_loop()


if __name__ == '__main__' :
    main()
    