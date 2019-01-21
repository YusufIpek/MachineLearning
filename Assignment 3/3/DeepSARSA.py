# coding: utf-8
# Deep SARSA

import ipympl
import matplotlib.pyplot as plt
import gym
import numpy as np
from tqdm import tqdm, trange # make your loops show a smart progress meter - just wrap any iterable loop
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
        ''' a feed forward network with only one linear layer. '''   
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)


def main_SARSA(env,epsilon = 0.3,gamma = 0.99,steps = 200,episodes = 5000,learning_rate = 0.001):
    '''
    SARSA algo:
    - Initialize parameters
    - Initialize Policy model and optimizers
    - for each episode
        - Initialize state S
        - choose action A using the policy
        - for each step in the episode
            - take a step with action A & get the reward R and next state S'
            - if next state S' is done and terminal
                - update the weights without using the Q_target
                - go to next episode
            - choose action A' for the next state S' using the policy
            - update the policy 
                update the weights with the Q_target 
            - update the action A = A' & the state S = S'
    '''
    env.seed(3333); torch.manual_seed(3333); np.random.seed(3333)

    # Initialize Parameters
    successful = []
    loss_history = []
    reward_history = []
    max_position = -0.4
    successes = 0
    position = []
    env._max_episode_steps = steps

    # Initialize Policy model and optimizers
    policy = Policy(env)
    loss_fn = nn.MSELoss()  # the mean squared error
    optimizer = optim.SGD(policy.parameters(), lr=learning_rate) # to optimize the parameters using SGD
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) # to adjust the learning rate
    first_succeeded_episode = -1

    for episode in trange(episodes): # trange is the range function but with ploting option
        # Initialize state S
        episode_loss = 0
        episode_reward = 0
        S = env.reset()

        # choose action A using the policy
        Q = policy(Variable(torch.from_numpy(S).type(torch.FloatTensor))) # return a tensor of the three actions with the value of choosing each one of them

        # act non-greedy or state-action have no value # exploration constant 
        rand_norm_uniform = np.random.uniform(0, 1)
        if rand_norm_uniform < epsilon:
           A = np.random.choice(env.action_space.n)
        else:
            _,A = torch.max(Q,-1)
            A = A.item()

        for s in range(steps):

            if (episode % 1000 == 0 and episode > 0):
                if s == 0 or s == 1999:
                    print('successful episodes: {}'.format(successes))
                #env.render()
            
            # act non-greedy or state-action have no value # exploration constant 
            rand_norm_uniform = np.random.uniform(0, 1)
            if rand_norm_uniform < epsilon:
                A = np.random.choice(env.action_space.n)
        
            # take a step with action A & get the reward R and next state S'  
            S_1, R, done, info = env.step(A)
            if done:
                if S_1[0] >= 0.5:
                    # On successful epsisodes, store the following parameters

                    # Adjust epsilon
                    #epsilon *= 0.99

                    # Adjust learning rate
                    scheduler.step()

                    # Store episode number if it is the first
                    if successes == 0:
                        first_succeeded_episode = episode

                    # Record successful episode
                    successes += 1
                
                    # Q_target = reward
                    Q_target = Q.clone()
                    Q_target = Variable(Q_target.data)
                    Q_target[A] = R

                    # Update policy
                    loss = loss_fn(Q,Q_target)
                    policy.zero_grad() # Zero the gradients before running the backward pass.
                    loss.backward()
                    optimizer.step()
                    # Record history
                    episode_loss += loss.item()
                    
                    # Keep track of max position
                    if S_1[0] > max_position:
                        max_position = S_1[0]

                episode_reward += R
                # Record history
                loss_history.append(episode_loss)
                reward_history.append(episode_reward)
                position.append(S_1[0])
                break # to terminate the episode
        

            # choose action A' for the next state S' using the policy
            Q_1 = policy(Variable(torch.from_numpy(S_1).type(torch.FloatTensor)))
            maxQ_1,A_1 = torch.max(Q_1,-1)
            
            # Create target Q value for training the policy
            Q_target = Q.clone()
            Q_target = Variable(Q_target.data)
            Q_target[A] = R + torch.mul(maxQ_1.detach(), gamma)

            # Calculate loss
            loss = loss_fn(Q, Q_target)
            
            # Update policy
            policy.zero_grad() # Zero the gradients before running the backward pass.
            loss.backward()
            optimizer.step()

            # Record history
            episode_loss += loss.item()
            episode_reward += R
            # Keep track of max position
            if S_1[0] > max_position:
                max_position = S_1[0]
            
            S = S_1
            A = A_1.item()
            Q = Q_1            

    print('successful episodes: {:d} - {:.4f}%'.format(successes, successes/episodes*100))
    print(" The first episode that reached the solution is: ",first_succeeded_episode)
    return policy, reward_history

def run_optimal_policy(env,policy,steps = 200,episodes = 100):
    # after finishing we want to test the policy
    success_counter = 0
    for iter_ in range(episodes):
        S = env.reset()
        # choose action A using the policy
        Q = policy(Variable(torch.from_numpy(S).type(torch.FloatTensor))) # return a tensor of the three actions with the value of choosing each one of them

        for s in range(steps):
            # env.render()
            
            # act greedy
            _,A = torch.max(Q,-1)
            A = A.item()
            # take a step with action A & get the reward R and next state S'  
            S_1, R, done, info = env.step(A)
            if done:
                if S_1[0] >= 0.5:
                    success_counter += 1
                    print(" **** successful try number {}  in testing phase, Car reached the goal. **** ".format(iter_))
                break # to terminate the episode

            # choose action A' for the next state S' using the policy
            Q_1 = policy(Variable(torch.from_numpy(S_1).type(torch.FloatTensor)))
            _,A_1 = torch.max(Q_1,-1)

            S = S_1
            A = A_1.item()
            Q = Q_1   
    print(" total succeeded {} out of {}, accuracy {}".format(success_counter,episodes,success_counter/episodes))           


def plot_average_reward(filename, df, average_reward_history):
    # print(df)
    pl = df.plot()
    pl.set_xlabel('Episodes')
    pl.set_ylabel('Avg Reward')
    plt.savefig(filename + '.png')
    plt.close() 

def compute_average(total_reward_history, steps):
    average_history = []
    total_reward_history = np.asarray(total_reward_history)
    total_reward_history = np.apply_along_axis(lambda x: x/steps, 1, total_reward_history)
    for i in range(total_reward_history.shape[1]):
        average_history.append(total_reward_history[:,i].mean())
    return average_history

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)

    episodes = 1000
    steps = 1000
    episode_attr = [x+1 for x in range(episodes)]
    runs = 2
    total_reward_history = []

    for run in range(runs):
        policy, reward_history = main_SARSA(env, steps=steps, episodes=episodes)
        total_reward_history.append(reward_history)
        run_optimal_policy(env,policy)

    average_reward_history = compute_average(total_reward_history, steps)
    df = pd.DataFrame(data={'avg reward':average_reward_history})    
    plot_average_reward("neural_network_sarsa_" + "steps" + str(steps) + "_episodes" + str(episodes),df, average_reward_history)
                  