import gym
import QL as tabularQL
import SARSA as SARSA
import DeepQL as DQL
from tile_sarsa import Semi_Episodic_SARSA 

def set_environment():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)
    return env    

if __name__ == '__main__':
    # TODO: set environment 
    env = set_environment()
    print(" (Task i): linear value-action function .... ")

    # Episodic algorithms
    
    # normal tabular q-learning algorithm
    ''' will not be used as it does not achieve any results'''
    print(" *** Trying a normal tabular Q-Learning agent:-- ")
    episodes = 10000
    print("Number of Episodes: ",episodes)
    tabularQL.main_QL(env,episodes = episodes)
    

    # Deep q-learning algorithm
    ''' will be used as our q-learning algorithm, managed to solve it after more than 2000 episode '''
    print("Number of Episodes: ",3000)
    print(" *** Now, Trying a linear Deep NN Q-Learning agent:-- ")
    policy = DQL.main_DQL(env)
    DQL.run_optimal_policy(env,policy)

    print(" *** Now, Trying a Deep NN-SARSA agent:-- ")
    policy = SARSA.main_SARSA(env)
    SARSA.run_optimal_policy(env,policy)

    # Continues algorithms
    print(" (Task ii): Non-linear value-action function approximator .... ")

    print(" *** Now, Trying a Semi-Episodic SARSA agent with tile coding:-- ")
    EpisodicSARSA = Semi_Episodic_SARSA(env)
    EpisodicSARSA.Semi_Episodic_SARSA()
    EpisodicSARSA.run_optimal_policy()
    print(" *** **** *** **** *** **** *** ")

    env.close()