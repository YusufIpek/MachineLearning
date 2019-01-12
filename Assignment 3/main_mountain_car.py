import gym
import QL as QL
import DQL as DQL


def set_environment():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)
    return env    

if __name__ == '__main__':
    # TODO: set environment 
    env = set_environment()

    # normal q-learning algorithm
    ''' will not be used as it does not achieve good results'''
    # QL.main_QL(env)


    # Deep q-learning algorithm
    ''' will be used as our q-learning algorithm, managed to solve it after more than 2000 episode '''
    DQL.main_DQL(env)

    # TODO: some plots and visualization
