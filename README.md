[//]: # (Image References)

[image1]: results.png "Training progess"
[image2]: tennis.gif "Trained agents"


# Project 3: Collaboration and Competition

## TL;DR;

This is a TD3 algorythm scaled to multi-agent enviroment. There's no shared parts between agents.
This is just 2 independent Agents training in same environment. 

To solve the envoromnent is it required to reach mean score of 0.5 in last 100 episodes. This solution reaches this in 2000 episodes. 

Trained agent looks like this:

![Trained Agent][image2]

# Solution

Just duplicating TD3 algorytmh from another project (link) wasn't enought. Agents didn't not trained or converged to edge cases. Sometimes one agent started to perform nicely, but another wasn't.

So, I spend few days running over hyperparamets. Seems, that final solution was to decrease learning rate to 1e-4 (strange for Adam, huh?) and also running training routine few times after every enovironment step.

Training progress:

![Trained Agent][image1]


# Potential improvments

 - Make shared critic 
 - Add BatchNorms to network architectures
 - Hyperopt chose of hyperparameters - seems this problem is quite sensitive to them

### Final set of hyperparameters

Learning routines per environment update = 3
Policy updated per agent train = 2
Batch Size = 512
Discount = 0.99
Replay buffer size = 1e5
Tau = 5e-3

### Networks architectures
Very small 64 neurons based:

#### Actor 
Relu activations in the middle, Tahn after last layer.
```
nn.Linear(state_dim, 64)
nn.Linear(64, 64)
nn.Linear(64, action_dim)
```

#### Critic
Relu activations in the middle, No activation after last one.
```
nn.Linear(state_dim + action_dim, 64)
nn.Linear(64, 64)
nn.Linear(64, 1)
```

# Description of files
 - actor[1,2].pth - trained agents. They can play agains each other and 1-1 2-2 as well
 - agent.py - naive TD3 agent
 - multiagent.py - wrapped to for fully separate feinforcement learning agents
 - noise.py - Ornstein-Uhlenbeck noise process
 - replay_buffer.py - Replay Buffer ;-)
 - Tennis.ipynb - training routine
 - Tennis_inference.py - demo of the trained agents

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.


In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started with project

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

## Acknowledgements

1. Replay Buffer module was taken from Baseline package from Open AI [https://github.com/openai/baselines]
2. TD3 is based on Medium explanaition [https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93]
3. My previous work for project 2 at Udacity [https://github.com/crazyleg/TD3-reacher]