import torch
import numpy as np
import pdb
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from networks import Actor, Critic

STEPS_PER_UPDATES = 3

class MultiAgent:
    def __init__(self, env, base_agent, n_agents=2, train_mode=True):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.train_mode = train_mode
        self.agents = [base_agent() for i in range(n_agents)]
        self.state = self.reset()
        
    def step(self):
        eee = [agent.select_action_with_noise(self.state,i) for i, agent in enumerate(self.agents)]
        actions = np.stack(eee)
        env_info = self.env.step(actions)[self.brain_name] 
        next_state = env_info.vector_observations
        reward = env_info.rewards                         # get reward (for each agent)
        done = env_info.local_done                        # see if episode finished
        self.total_score += np.array(reward)
        
        self.done = done
        
        for i, agent in enumerate(self.agents):
            agent.replay_buffer.add(self.state, eee, reward, next_state, done)
        
        self.state = next_state
        
        for i in range(STEPS_PER_UPDATES):
            [agent.step(i) for i, agent in enumerate(self.agents)]
        
    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        state = env_info.vector_observations
        
        self.total_score = np.zeros(2)
        self.policy_freq_it = 0
        
        [agent.reset() for agent in self.agents]
        
        return state