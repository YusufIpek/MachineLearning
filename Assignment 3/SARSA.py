# coding: utf-8
# Deep Q-Learning

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
from tensorboardX import SummaryWriter
from datetime import datetime
import glob, os

# ## Policy
# We use a shallow neural network with 300 hidden units to learn our policy.

class Policy(nn.Module):
    def __init__(self,env):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 300
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)
    
    def forward(self, x): 
        ''' a feed forward network with only one linear layer. '''   
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)


def main_SARSA(env):
    '''
    SARSA algo:
    - Initialize parameters
    - Initialize Policy model and optimizers
    - for each episode
        - Initialize state S
        - choose action A using the policy
        - for each step in the episode
            - take a step with action A & get the reward R and next state S'
            - choose action A' for the next state S' using the policy
            - update the policy 
                q[S,A] = q[S,A] + lr*(R + gamma*q[S',A'] - q[S,A] )
            - update the action A = A' & the state S = S'
    '''
    env.seed(3333); torch.manual_seed(3333); np.random.seed(3333)

    # SummaryWriter is a high-level api to create an event file in a given directory and add summaries and events to it.
    writer = SummaryWriter('~/tboardlogs/{}'.format(datetime.now().strftime('%b%d_%H-%M-%S')))

    # Initialize Parameters
    successful = []
    steps = 2000
    S = env.reset()
    epsilon = 0.1
    gamma = 0.99
    loss_history = []
    reward_history = []
    episodes = 3000
    max_position = -0.4
    learning_rate = 0.001
    successes = 0
    position = []

    # Initialize Policy model and optimizers
    policy = Policy(env)
    loss_fn = nn.MSELoss()  # the mean squared error
    optimizer = optim.SGD(policy.parameters(), lr=learning_rate) # to optimize the parameters using SGD
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99) # to adjust the learning rate
    first_succeeded_episode = -1

    for episode in trange(episodes): # trange is the range function but with ploting option
        # Initialize state S
        episode_loss = 0
        episode_reward = 0
        S = env.reset()

        # choose action A using the policy
        Q = policy(Variable(torch.from_numpy(S).type(torch.FloatTensor)))

        # act non-greedy or state-action have no value # exploration constant 
        if np.random.rand(1) < epsilon:
           A = np.random.randint(0,3)
        else:
            _,A = torch.max(Q,-1)
            A = A.item()

        for s in range(steps):
            # Uncomment to render environment
            if episode % 1000 == 0 and episode > 0:
                if s == 0:
                    print('successful episodes: {}'.format(successes))
                env.render()
            
            # act non-greedy or state-action have no value # exploration constant 
            if np.random.rand(1) < epsilon:
                A = np.random.randint(0,3)
        
            # take a step with action A & get the reward R and next state S'  
            S_1, R, done, info = env.step(A)

            if done:
                if S_1[0] >= 0.5:
                    # On successful epsisodes, store the following parameters

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
                

                    
                    # Update policy
                    loss = loss_fn(Q, Q)
                    policy.zero_grad() # Zero the gradients before running the backward pass.
                    loss.backward()
                    optimizer.step()
                    # Record history
                    episode_loss += loss.item()
                    episode_reward += R
                    # Keep track of max position
                    if S_1[0] > max_position:
                        max_position = S_1[0]
                        writer.add_scalar('data/max_position', max_position, episode)

                elif S_1[0] < 0.5:
                    writer.add_scalar('data/success', 0, episode)
                
                # Record history
                loss_history.append(episode_loss)
                reward_history.append(episode_reward)
                writer.add_scalar('data/episode_loss', episode_loss, episode)
                writer.add_scalar('data/episode_reward', episode_reward, episode)
                weights = np.sum(np.abs(policy.l2.weight.data.numpy()))+np.sum(np.abs(policy.l1.weight.data.numpy()))
                writer.add_scalar('data/weights', weights, episode)
                writer.add_scalar('data/position', S_1[0], episode)
                position.append(S_1[0])

                break # to terminate the episode
        

            # choose action A' for the next state S' using the policy
            Q_1 = policy(Variable(torch.from_numpy(S_1).type(torch.FloatTensor)))
            _,A_1 = torch.max(Q_1,-1)
            
            # Create target Q value for training the policy
            Q_target = Q.clone()
            Q_target = Variable(Q_target.data)
            Q_target[A] = R + torch.mul(A_1.detach(), gamma)

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
                writer.add_scalar('data/max_position', max_position, episode)
            
            S = S_1
            A_1 = A_1.item()
            A = A_1
            Q = Q_1            

                


    writer.close()
    print('successful episodes: {:d} - {:.4f}%'.format(successes, successes/episodes*100))
    print(" The first episode that reached the solution is: ",first_succeeded_episode)


'''
# ## Plot Results
# Around episode 1000 the agent begins to successfully complete episodes.

plt.figure(2, figsize=[10,5])
p = pd.Series(position)
ma = p.rolling(10).mean()
plt.plot(p, alpha=0.8)
plt.plot(ma)
plt.xlabel('Episode')
plt.ylabel('Position')
plt.title('Car Final Position')
plt.savefig('Final Position.png')
plt.show()


# <iframe width="900" position="800" frameborder="0" scrolling="no" src="//plot.ly/~ts1829/22.embed"></iframe>

# ## Visualize Policy
# We can see the policy by plotting the agent’s choice over a combination of positions and velocities. You can see that the agent learns to, *usually*, move left when the car’s velocity is negative and then switch directions when the car’s velocity becomes positive with a few position and velocity combinations on the left side of the environment where the agent will do nothing.

X = np.random.uniform(-1.2, 0.6, 10000)
Y = np.random.uniform(-0.07, 0.07, 10000)
Z = []
for i in range(len(X)):
    _, temp = torch.max(policy(Variable(torch.from_numpy(np.array([X[i],Y[i]]))).type(torch.FloatTensor)), dim =-1)
    z = temp.item()
    Z.append(z)
Z = pd.Series(Z)
colors = {0:'blue',1:'lime',2:'red'}
colors = Z.apply(lambda x:colors[x])
labels = ['Left','Right','Nothing']

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
fig = plt.figure(3, figsize=[7,7])
ax = fig.gca()
plt.set_cmap('brg')
surf = ax.scatter(X,Y, c=Z)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_title('Policy')
recs = []
for i in range(0,3):
     recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
plt.legend(recs,labels,loc=4,ncol=3)
fig.savefig('Policy.png')
plt.show()

'''