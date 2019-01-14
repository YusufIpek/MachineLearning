import gym
import QL as QL
import DQL as DQL
#import nonlinearDQL as nonlinearDQL

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
    print(" *** Now, Trying a linear Deep Q-Learning agent:-- ")
    print("Number of Episodes: ",3000)
    DQL.main_DQL(env)

    # TODO: some plots and visualization

    print(" *** **** *** **** *** **** *** ")
    # Continues algorithms

    # Deep q-learning 
    print(" (Task ii): Non-linear value-action function approximator .... ")
    #nonlinearDQL.main_fun(env)

