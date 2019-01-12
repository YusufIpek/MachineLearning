# coding: utf-8
# # Mountain Car v0 - Deep Q-Learning

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


'''
# Running the environment with random actions produces no successful episodes in a run of 1000 episodes.  


max_position = -.4
positions = np.ndarray([0,2])
rewards = []
successful = []
for episode in range(1000):
    running_reward = 0
    env.reset()
    done = False
    for i in range(200):
        state, reward, done, _ = env.step(np.random.randint(0,3))
        # Give a reward for reaching a new maximum position
        if state[0] > max_position:
            max_position = state[0]
            positions = np.append(positions, [[episode, max_position]], axis=0)
            running_reward += 10
        else:
            running_reward += reward
        if done: 
            if state[0] >= 0.5:
                successful.append(episode)
            rewards.append(running_reward)
            break

print('Furthest Position: {}'.format(max_position))
plt.figure(1, figsize=[10,5])
plt.subplot(211)
plt.plot(positions[:,0], positions[:,1])
plt.xlabel('Episode')
plt.ylabel('Furthest Position')
plt.subplot(212)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
print('successful episodes: {}'.format(np.count_nonzero(successful)))
'''

# ## Policy
# We use a shallow neural network with 200 hidden units to learn our policy.

class Policy(nn.Module):
    def __init__(self,env):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 200
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)
    
    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)
    


# We use the Q-Learning update equation to update our action value function based on the reward for the agent's action and the maximum future action value function one step in the future.  The portion inside the brackets becomes the loss function for our neural network where $Q(s_t,a_t)$ is the output of our network and $ r_t + \gamma \max\limits_{a} Q(s_{t+1},a_{t+1}) $ is the target Q value as well as the label for our neural net turning the problem into a supervised learning problem.
# 
# $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \big[r_t + \gamma \max\limits_{a} Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\big]  \tag{Q-Learning}$$

def main_DQL(env):
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
                
            
            # Get first action value function
            Q = policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))
            
            # Choose epsilon-greedy action
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0,3)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
            
            # Step forward and receive next state and reward
            state_1, reward, done, _ = env.step(action)
            
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