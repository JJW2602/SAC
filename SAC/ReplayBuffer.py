import numpy as np

class ReplayBuffer:
    def __init__(self,max_capacity):
        self.max_size = max_capacity
        self.size=0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None

    def store(self, obs : np.ndarray, action, reward, next_obs, done):
        
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.empty((self.max_size, *obs.shape),dtype=obs.dtype)
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape),dtype=reward.dtype)
            self.next_observations = np.empty((self.max_size, *next_obs.shape), dtype=next_obs.dtype)
            self.dones = np.empty((self.max_size, *done.shape),dtype=done.dtype)
            
        self.observations[self.size % self.max_size] = obs
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_obs
        self.dones[self.size % self.max_size]=done
        self.size+=1

    def sample(self, sample_batch_size):
        random_indices = (np.random.randint(0, self.size, size = (sample_batch_size,)) % self.max_size)
        return {
            "observations": self.observations[random_indices],
            "actions": self.actions[random_indices],
            "rewards": self.rewards[random_indices],
            "next_observations": self.next_observations[random_indices],
            "dones" : self.dones[random_indices]
        }

    
    