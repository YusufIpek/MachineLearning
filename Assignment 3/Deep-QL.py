# coding: utf-8
# Deep Q-Learning

import ipympl
import matplotlib.pyplot as plt
import gym
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
import glob, os


# ## Policy
# We use a shallow neural network with 200 hidden units to learn our policy.

class Policy(nn.Module):
    def __init__(self,env):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 300
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)
    
    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)
    
def main_DQL(env):
    '''
    D-QL algo:
    - Initialize parameters
    - Initialize Policy model and optimizers
    - for each episode
        - Initialize state S
        - for each step in the episode
            - choose action A using the policy
            - take a step with action A & get the reward R and next state S'
            - choose action A' for the next state S' using the policy
            - update the policy 
                update the weights with the Q_target 
            - update the action A = A' & the state S = S'
    '''
    env.seed(1); torch.manual_seed(1); np.random.seed(1)
    writer = SummaryWriter('~/tboardlogs/{}'.format(datetime.now().strftime('%b%d_%H-%M-%S')))

    # Parameters
    successful = []
    steps = 2000
    state = env.reset()
    epsilon = 0.3
    gamma = 0.99
    loss_history = []
    reward_history = []
    episodes = 3000
    max_position = -0.4
    learning_rate = 0.001
    successes = 0
    position = []

    # Initialize Policy
    policy = Policy(env)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    first_succeeded_episode = -1

    for episode in trange(episodes):
        episode_loss = 0
        episode_reward = 0
        state = env.reset()
        for s in range(steps):
            # Uncomment to render environment
            if episode % 1000 == 0 and episode > 0:
                if s == 0:
                    print('successful episodes: {}'.format(successes))
                env.render()
                
            
            # choose action A using the policy
            Q = policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))
            
            # Choose epsilon-greedy action
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0,3)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
            
            # take a step with action A & get the reward R and next state S'
            state_1, reward, done, info = env.step(action)
            
            # Find max Q for t+1 state
            Q1 = policy(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
            maxQ1, _ = torch.max(Q1, -1)
            
            # Create target Q value for training the policy
            Q_target = Q.clone()
            Q_target = Variable(Q_target.data)
            Q_target[action] = reward + torch.mul(maxQ1.detach(), gamma)
            
            # Calculate loss
            loss = loss_fn(Q, Q_target)
            
            # Update policy
            policy.zero_grad()
            loss.backward()
            optimizer.step()

            # Record history
            episode_loss += loss.item()
            episode_reward += reward
            # Keep track of max position
            if state_1[0] > max_position:
                max_position = state_1[0]
                writer.add_scalar('data/max_position', max_position, episode)
            
            if done:
                if state_1[0] >= 0.5:
                    # On successful epsisodes, adjust the following parameters

                    # Adjust epsilon
                    epsilon *= .99
                    writer.add_scalar('data/epsilon', epsilon, episode)

                    # Adjust learning rate
                    scheduler.step()
                    writer.add_scalar('data/learning_rate', optimizer.param_groups[0]['lr'], episode)

                    # Store episode number if it is the first
                    if successes == 0:
                        first_succeeded_episode = episode

                    # Record successful episode
                    successes += 1
                    writer.add_scalar('data/cumulative_success', successes, episode)
                    writer.add_scalar('data/success', 1, episode)
                
                elif state_1[0] < 0.5:
                    writer.add_scalar('data/success', 0, episode)
                
                # Record history
                loss_history.append(episode_loss)
                reward_history.append(episode_reward)
                writer.add_scalar('data/episode_loss', episode_loss, episode)
                writer.add_scalar('data/episode_reward', episode_reward, episode)
                weights = np.sum(np.abs(policy.l2.weight.data.numpy()))+np.sum(np.abs(policy.l1.weight.data.numpy()))
                writer.add_scalar('data/weights', weights, episode)
                writer.add_scalar('data/position', state_1[0], episode)
                position.append(state_1[0])

                break
            else:
                state = state_1
                
    writer.close()
    print('successful episodes: {:d} - {:.4f}%'.format(successes, successes/episodes*100))
    print(" The first episode that reached the solution is: ",first_succeeded_episode)

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)
    main_DQL(env)