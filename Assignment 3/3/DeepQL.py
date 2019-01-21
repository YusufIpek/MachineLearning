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
from datetime import datetime
import glob, os


# ## Policy
# We use a shallow neural network with 500 hidden units to learn our policy.

class Policy(nn.Module):
    def __init__(self,env,hidden_units = 500):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = hidden_units
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
    env.seed(3333); torch.manual_seed(3333); np.random.seed(3333)
    # Parameters
    successful = []
    steps = 200
    state = env.reset()
    epsilon = 0.2
    gamma = 0.99
    loss_history = []
    reward_history = []
    episodes = 5000
    max_position = -0.4
    learning_rate = 0.001
    successes = 0
    position = []
    env._max_episode_steps = steps
    
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
            
            if episode % 1000 == 0 and episode > 0:
                if s == 0 or s == 1999:
                    print('successful episodes: {}'.format(successes))
                #env.render()
                
            
            # choose action A using the policy
            Q = policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))
            
            # Choose epsilon-greedy action
            rand_norm_uniform = np.random.uniform(0, 1)
            if rand_norm_uniform < epsilon:
                action = np.random.choice(env.action_space.n)
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
            
            if done:
                if state_1[0] >= 0.5:
                    # On successful epsisodes, adjust the following parameters

                    # Adjust epsilon
                    #epsilon *= .99

                    # Adjust learning rate
                    scheduler.step()

                    # Store episode number if it is the first
                    if successes == 0:
                        first_succeeded_episode = episode

                    # Record successful episode
                    successes += 1
                
                # Record history
                loss_history.append(episode_loss)
                reward_history.append(episode_reward)
                position.append(state_1[0])

                break
            else:
                state = state_1

    print('successful episodes: {:d} - {:.4f}%'.format(successes, successes/episodes*100))
    print(" The first episode that reached the solution is: ",first_succeeded_episode)
    return policy

def run_optimal_policy(env,policy,steps = 1000,episodes = 100):
    # after finishing we want to test the policy
    env._max_episode_steps = steps
    success_counter = 0
    for iter_ in range(episodes):
        S = env.reset()
        
        for s in range(steps):
            #env.render()
            # choose action A using the policy
            Q = policy(Variable(torch.from_numpy(S).type(torch.FloatTensor))) # return a tensor of the three actions with the value of choosing each one of them
    
            # act greedy
            _,A = torch.max(Q,-1)
            # take a step with action A & get the reward R and next state S'  
            S, R, done, info = env.step(A.item())
            if done:
                if S[0] >= 0.5:
                    success_counter +=1
                    print(" **** successful try number {}  in testing phase, Car reached the goal. **** ".format(iter_ +1))
                break # to terminate the episode          

    print(" total succeeded {} out of {}, accuracy {}".format(success_counter,episodes,success_counter/episodes))     

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)
    policy = main_DQL(env)
    run_optimal_policy(env,policy)