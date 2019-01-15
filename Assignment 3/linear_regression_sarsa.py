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

def predict(x, weights):
    res = []
    i = 0
    val1 = 0
    val2 = 0
    for w in weights:  
        val1 += w[0] * x[0]**i
        val2 += w[1] * x[1]**i
        i += 1
    res.append(val1)
    res.append(val2)
    return res


def target_value(x, weights, gamma, R):
    predicted_value = predict(x,weights)
    return [(predicted_value[0] * gamma) + R, (predicted_value[1] * gamma) + R]

def cost_function(x, target, weights):
    N = len(target)
    prediction = predict(x, weights)
    sq_error = [(prediction[0] - target[0])**2,(prediction[1] - target[1])**2]
    return [1.0/(2*N) * sq_error[0], 1.0/(2*N) * sq_error[1]]

def update_weights_vectorized(x, target, weights, learning_rate):
    # See: https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
    prediction = predict(x, weights)
    error = [target[0] - prediction[0], target[1] - prediction[1]]
    gradient = [-x[0]*error[0], -x[1]*error[1]]

    gradient = [gradient[0] * learning_rate, gradient[1] * learning_rate]
    updated_weights = list(map(lambda w: [w[0]-gradient[0],w[1]-gradient[1]], weights))
    return updated_weights



def main_SARSA(env,epsilon = 0.2,gamma = 0.99,steps = 2000,episodes = 3000,learning_rate = 0.001):
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

    # Initialize Policy model and optimizers
    # 
    first_succeeded_episode = -1
    weights = []
    for i in range(200):
        weights.append([1,1])
    for episode in trange(episodes): # trange is the range function but with ploting option
        # Initialize state S
        episode_loss = 0
        episode_reward = 0
        S = env.reset()

        
        # act non-greedy or state-action have no value # exploration constant 
        rand_norm_uniform = np.random.uniform(0, 1)
        if rand_norm_uniform < epsilon:
           A = np.random.choice(env.action_space.n)
        else:
            # choose action A using the policy
            A = predict(S,weights)
            
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
                    epsilon *= 0.99

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
                    episode_reward += R
                    # Keep track of max position
                    if S_1[0] > max_position:
                        max_position = S_1[0]
                
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
    return policy

def run_optimal_policy(env,policy,steps = 2000,episodes = 10):
    # after finishing we want to test the policy
    success_counter = 0
    for iter_ in range(episodes):
        S = env.reset()
        # choose action A using the policy
        Q = policy(Variable(torch.from_numpy(S).type(torch.FloatTensor))) # return a tensor of the three actions with the value of choosing each one of them

        for s in range(steps):
            env.render()
            
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
    print(" total succeeded {} out of {}".format(success_counter,episodes))         

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)
    policy = main_SARSA(env)
    run_optimal_policy(env,policy)                