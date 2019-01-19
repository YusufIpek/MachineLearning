# coding: utf-8
# semi-episodic SARSA

import gym
import numpy as np
from tqdm import trange # make your loops show a smart progress meter - just wrap any iterable loop
from TileCoding import *
import csv
import pandas as pd

class Semi_Episodic_SARSA:
    def __init__(self,env,weights = None,max_tiles = 2048,num_tilings = 8,features_type = True):
        self.env = env
        self.maxtiles = max_tiles
        self.numtilings = num_tilings
        if weights == None:
            self.weights = np.zeros(self.maxtiles)
        else:
            self.weights = weights
        self.max_position, self.max_velocity = tuple(self.env.observation_space.high)
        self.min_position, self.min_velocity = tuple(self.env.observation_space.low)
        
        self.hashTable = IHT(self.maxtiles)
        #self.features_type = features_type #set True for tile features, set false for normal derivation
        #ploynomial features


    def Semi_Episodic_SARSA(self,epsilon = 0.2,gamma = 0.99,steps = 2000,episodes = 500,learning_rate = 0.001):
        '''
        SARSA algo:
        - Initialize parameters
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
        self.env.seed(3333)
        np.random.seed(3333)
        # Initialize Parameters
        self.env._max_episode_steps = steps
        successes = 0
        position = []
        first_succeeded_episode = -1
        reward_history = []
        for episode in trange(episodes): # trange is the range function but with ploting option
            # Initialize state S
            episode_reward = 0
            S = self.env.reset()
            # act non-greedy or state-action have no value # exploration constant 
            if np.random.uniform(0, 1) < epsilon:
                A = np.random.choice(self.env.action_space.n)
            else:
                # TODO: take action from the policy
                A = self.take_action(S)     

            for s in range(steps):
                if (episode % 100 == 0 and episode > 0):
                    if s == 0 or s == 1999:
                        print('successful episodes: {}'.format(successes))
                    #env.render()
                
                # act non-greedy or state-action have no value # exploration constant 
                if np.random.uniform(0, 1) < epsilon:
                    A = np.random.choice(self.env.action_space.n)
            
                # take a step with action A & get the reward R and next state S'  
                S_1, R, done, info = self.env.step(A)
                if done:
                    if S_1[0] >= 0.5:
                        # On successful epsisodes, store the following parameters
                        
                        # Adjust epsilon
                        epsilon *= 0.99

                        # Store episode number if it is the first
                        if successes == 0:
                            first_succeeded_episode = episode

                        # Record successful episode
                        successes += 1
                    
                        #TODO: update your weights
                        change = learning_rate * (R - self.build_q_fun(S, A))
                        self.weights += change * np.array(self.delta(S, A))
                        weights = self.weights

                    episode_reward += R
                    # Record history
                    reward_history.append(episode_reward)
                    position.append(S_1[0])
                    break # to terminate the episode
            
                # choose action A' for the next state S' using the policy
                #TODO: choose second action A'
                A_1 = self.take_action(S_1)
                
                # Create target Q value for training the policy
                #TODO: create the q_target value
                qdash = self.build_q_fun(S_1, A_1)
                q = self.build_q_fun(S, A)
                # Update policy
                #TODO: update policy
                q_target = R + gamma * qdash
                change = learning_rate * (q_target - q)
                self.weights += change * np.array(self.delta(S, A))
                weights = self.weights
                
                # Record history
                episode_reward += R
                
                #TODO: calculate S & A 
                S = S_1
                A = A_1
        # overall_reward_history.append(reward_history)
        # self.store_reward(overall_reward_history, runs, episodes=episodes)
        print('successful episodes: {:d} - {:.4f}%'.format(successes, successes/episodes*100))
        print(" The first episode that reached the solution is: ",first_succeeded_episode)
        return reward_history

    def store_reward(self, df, reward_history, run, steps=500, episodes=500):
        average_reward = ([str(x/steps).replace('.',',') for x in reward_history])  
        df['avg reward run ' + str(run)] = average_reward    
        df.to_csv('./benchmark_tile_sarsa_steps' + str(steps) + "_episodes" + str(episodes) + ".csv", sep=' ', index=False)
        

    # get indices of active tiles for given state and action
    def getActiveTiles(self,position, velocity, action):
        activeTiles = tiles(self.hashTable, self.numtilings,
                            [self.numtilings * position / (self.max_position - self.min_position), self.numtilings * velocity / (self.max_velocity - self.min_velocity)],
                            [action])
        return activeTiles

    def take_action(self,State):
        ''' take a state observation, returns the best chosen action.'''
        actions_list = [self.build_q_fun(State, action) for action in range(self.env.action_space.n)]
        return np.argmax(actions_list)

    # q(S,A,weights) = x(S,A).T weights
    def build_q_fun(self,State, action):
        q_temp = np.matmul(self.get_tile_features(State, action), self.weights)
        return q_temp
   

    # delta_q(S,A,weights) = x(S,A)
    def delta(self,State, action):
        delta_val = self.get_tile_features(State, action)
        return delta_val

    def get_tile_features(self,State, action):
        tileIndices = self.getActiveTiles(State[0], State[1], action)
        feature = [0] * self.maxtiles
        for tile_index in tileIndices:
            feature[tile_index] = 1
        return feature

    def run_optimal_policy(self,steps = 2000,episodes = 10):
        # after finishing we want to test the policy
        success_counter = 0
        for iter_ in range(episodes):
            S = self.env.reset()
            # choose action A using the policy
            A = self.take_action(S) 
            #env = wrappers.Monitor(env, './semi-sarsa', force=True)  # to take a video snapshot
            for s in range(steps):
                #self.env.render()

                # take a step with action A & get the reward R and next state S'  
                S_1, R, done, info = self.env.step(A)
                if done:
                    if S_1[0] >= 0.5:
                        success_counter += 1
                        print(" **** successful try number {}  in testing phase, Car reached the goal. **** ".format(iter_))
                    break # to terminate the episode

                # choose action A' for the next state S' using the policy
                A = self.take_action(S_1)
                S = S_1

        print(" total succeeded {} out of {}".format(success_counter,episodes))         

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)

    episodes = 500
    steps = 1000
    episode_attr = [x+1 for x in range(episodes)]
    df = pd.DataFrame(data={'Episodes':episode_attr}) #create first column which represents the episodes
    runs = 3
    for run in range(runs):
        EpisodicSARSA = Semi_Episodic_SARSA(env,features_type = True)
        reward_history = EpisodicSARSA.Semi_Episodic_SARSA(steps=steps, episodes=episodes)
        EpisodicSARSA.store_reward(df, reward_history, run, steps=steps, episodes=episodes)
        EpisodicSARSA.run_optimal_policy() 
