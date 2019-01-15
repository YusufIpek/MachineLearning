import gym
import QL as QL
import SARSA as SARSA
import DeepQL as DQL

def set_environment():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)
    return env    

if __name__ == '__main__':
    # TODO: set environment 
    env = set_environment()
    print(" (Task i): linear value-action function approximator .... ")

    # Episodic algorithms
    """
    # normal q-learning algorithm
    ''' will not be used as it does not achieve good results'''
    print(" *** Trying a normal linear Q-Learning agent:-- ")
    episodes = 10000
    print("Number of Episodes: ",episodes)
    QL.main_QL(env,episodes = episodes)
    """

    # Deep q-learning algorithm
    ''' will be used as our q-learning algorithm, managed to solve it after more than 2000 episode '''
    print("Number of Episodes: ",3000)
    # print(" *** Now, Trying a linear Deep Q-Learning agent:-- ")
    # DQL.policy = main_DQL(env)
    # DQL.run_optimal_policy(env,policy)

    print(" *** Now, Trying a linear SARSA agent:-- ")
    policy = SARSA.main_SARSA(env)
    SARSA.run_optimal_policy(env,policy)
    # TODO: some plots and visualization

    print(" *** **** *** **** *** **** *** ")
    # Continues algorithms

    # Deep q-learning 
    print(" (Task ii): Non-linear value-action function approximator .... ")
    # 

