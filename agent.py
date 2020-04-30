import torch
import numpy as np
import random
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from networks import Actor, Critic
from noise import OUNoise

class TD3MultiAgent:
    def __init__(self):
      
        self.max_action = 1
        self.policy_freq = 2
        self.policy_freq_it = 0
        self.batch_size = 512
        self.discount = 0.99
        self.replay_buffer = int(1e5)
        
        
        self.device = 'cuda'
        
        self.state_dim = 24
        self.action_dim = 2
        self.max_action = 1
        self.policy_noise = 0.1
        self.agents = 1
        
        self.random_period = 1e4
        
        self.tau = 5e-3
        
        self.replay_buffer = ReplayBuffer(self.replay_buffer)
        
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
#         self.actor.load_state_dict(torch.load('actor2.pth'))
#         self.actor_target.load_state_dict(torch.load('actor2.pth'))

        self.noise = OUNoise(2, 32)
        
        
        self.critic = Critic(48, self.action_dim).to(self.device)
        self.critic_target = Critic(48, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    
    def select_action_with_noise(self, state, i):
        import pdb
        ratio = len(self.replay_buffer)/self.random_period

        if len(self.replay_buffer)>self.random_period:
            
            state = torch.FloatTensor(state[i,:]).to(self.device)
            action = self.actor(state).cpu().data.numpy()

            if self.policy_noise != 0: 
                action = (action + self.noise.sample())
            return action.clip(-self.max_action,self.max_action)
        
        else:
            q= self.noise.sample()
            return q
   
    
    def step(self, i):
        if len(self.replay_buffer)>self.random_period/2:
            # Sample mini batch
#         if True:
            import pdb
            s, a, r, s_, d = self.replay_buffer.sample(self.batch_size)
            
            state = torch.FloatTensor(s[:,i,:]).to(self.device)
            action = torch.FloatTensor(a[:,i,:]).to(self.device)
            next_state = torch.FloatTensor(s_[:,i,:]).to(self.device)
            
            a_state = torch.FloatTensor(s).to(self.device).reshape(-1,48)
            a_action = torch.FloatTensor(a).to(self.device).reshape(-1,4)
            a_next_state = torch.FloatTensor(s_).to(self.device).reshape(-1,48)
            
            done = torch.FloatTensor(1 - d[:,i]).to(self.device)
            reward = torch.FloatTensor(r[:,i]).to(self.device)
#             pdb.set_trace()
            # Select action with the actor target and apply clipped noise
            noise = torch.FloatTensor(a[:,i,:]).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-0.1,0.1) # NOISE CLIP WTF?
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            # Compute the target Q value

            target_Q1, target_Q2 = self.critic_target(a_next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.reshape(-1,1) + (done.reshape(-1,1) * self.discount * target_Q).detach()
            
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(a_state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.policy_freq_it % self.policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(a_state, self.actor(state)).mean()
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


            self.policy_freq_it += 1
        
        return True
        
    
    def reset(self):
        self.policy_freq_it = 0
        self.noise.reset()