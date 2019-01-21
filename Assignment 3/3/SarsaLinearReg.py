# coding: utf-8
# semi-episodic SARSA

import gym
import numpy as np
from tqdm import trange # make your loops show a smart progress meter - just wrap any iterable loop
import random

class Linear_Reg_SARSA:
    def __init__(self,env,weights = None,max_features = 2048,features_type = True,basis_type = True):
        self.env = env
        # set up features function
        self.features = []
        self.features_type = features_type
        self.basis_type = basis_type
        self.max_features = max_features
        self.set_features()
        if weights == None:
            self.weights = np.zeros(self.max_features)
        else:
            self.weights = weights
        #self.max_position, self.max_velocity = tuple(self.env.observation_space.high)
        #self.min_position, self.min_velocity = tuple(self.env.observation_space.low)

    def set_features(self):
        if self.basis_type == True:
            if self.features_type == False:
                self.max_features = 1801
                for i in range(self.max_features):
                    self.features.append([0,0])
                self.mapper = {} # dict contain the mapped value
                pos = 0
                for i in range(-1200,601):
                    #self.mapper.update({round( (i/100)*(j/200),6):pos})
                    self.mapper.update({round( (i/1000),5):pos})
                    pos+=1
                #print(pos)

            else:# trying S power j 
                for i in range(0, self.max_features):
                    #k = i if i <= 2 else 2
                    self.features.append(lambda s, i= i : pow(s, i)) 
            
        else:# trying fourier basis 
            for i in range(0, self.max_features):
                self.features.append(lambda s, i=i: np.cos(i * np.pi * s)) 

    def Linear_Reg_SARSA(self,epsilon = 0.2,gamma = 0.99,steps = 2000,episodes = 500,learning_rate = 0.001):
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
        self.env._max_episode_steps = 1000
        reward_history = []
        successes = 0
        position = []
        first_succeeded_episode = -1

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
                if (episode % 50 == 0 and episode > 0):
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
                        delta = self.get_linear_features(S)
                        self.weights += np.dot(delta,change)

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
                delta = self.get_linear_features(S)
                self.weights += np.dot(delta,change)

                # Record history
                episode_reward += R
                
                #TODO: calculate S & A 
                S = S_1
                A = A_1          

        print('successful episodes: {:d} - {:.4f}%'.format(successes, successes/episodes*100))
        print(" The first episode that reached the solution is: ",first_succeeded_episode)

    def take_action(self,State):
        ''' take a state observation, returns the best chosen action.'''
        temp = []
        actions_list = [self.build_q_fun(State, action) for action in range(0,3)]
        for a_list in actions_list:
            temp.append(a_list[0])
        return np.argmax(temp)

    # q(S,A,weights) = weights.T features(S,A)
    def build_q_fun(self,State, action):
        return np.dot(self.weights,self.get_linear_features(State) )

    def get_linear_features(self,state):
        if self.basis_type == True:
            if self.features_type == True:
                a_feature = []
                for i in range(0,self.max_features):
                    a_feature.append(pow(state, random.choice([0,1,i]) ) )
                return np.asarray(a_feature)
            else:
                return self.get_special_activities(state)
        else:   
            return np.asarray([func(state) for func in self.features])
    

    def get_special_activities(self,state,n=1,k = 2):
        ''' mapping the states
        TODO: create state features small list
        TODO: create feature list of zeros 
        TODO: map each state to specific four elements in the feature list
        TODO: update the feature list with these elements
        return the new feature list
        '''
        mappp = self.mapper
        length = (n+1)**k
        for i in range(1,length ):
            if state[0] == 0:state[0]+=0.01 
            if state[1] == 0:state[1]+=0.01 
            s_feature = pow(state,i)
            key = round( (s_feature[0]/1000),3 )
            if key in self.mapper :
                self.features[self.mapper[key]] = s_feature
        return self.features

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

        print(" total succeeded {} out of {}, accuracy {}".format(success_counter,episodes,success_counter/episodes))           

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)    
    EpisodicSARSA = Linear_Reg_SARSA(env,features_type = False,basis_type = True) # for fourier basis_type = False, if feature_type is true choose small features randomly, else normal features 
    EpisodicSARSA.Linear_Reg_SARSA()
    EpisodicSARSA.run_optimal_policy() 
