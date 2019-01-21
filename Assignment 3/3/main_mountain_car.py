import gym
import tabularQL as tabularQL
import tabularSARSA as tabularSARSA
import DeepSARSA as SARSA
import DeepQL as DQL
from tile_sarsa import Semi_Episodic_SARSA 
from SarsaLinearReg import Linear_Reg_SARSA

def set_environment():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)
    return env    

if __name__ == '__main__':
    # TODO: set environment 
    env = set_environment()
    
    print(" (Task i):  .... ")

    """
    # normal tabular q-learning algorithm
    ''' will not be used as it does not achieve any results'''
    print(" *** Trying a normal tabular SARSA agent:-- ")
    episodes = 10000
    print("Number of Episodes: ",episodes)
    tabularSARSA.main_SARSA(env,episodes = episodes)


    print(" *** Trying a normal tabular Q-Learning agent:-- ")
    episodes = 10000
    print("Number of Episodes: ",episodes)
    tabularQL.main_QL(env,episodes = episodes)
    
    # Episodic algorithms
    print(" *** Now, Trying linear regression SARSA using a normal basis method for features construction:--")
    EpisodicSARSA = Linear_Reg_SARSA(env,basis_type = True,features_type = False)
    EpisodicSARSA.Linear_Reg_SARSA()
    EpisodicSARSA.run_optimal_policy() 


    print(" *** Now, Trying linear regression SARSA using a fourier basis method for features construction:--")
    EpisodicSARSA = Linear_Reg_SARSA(env,basis_type = False)
    EpisodicSARSA.Linear_Reg_SARSA()
    EpisodicSARSA.run_optimal_policy() 
    

    # Deep learning algorithms
    ''' will be used as our q-learning algorithm, managed to solve it after more than 2000 episode '''
    print("Number of Episodes: ",5000)
    print(" *** Now, Trying a linear Deep NN Q-Learning agent:-- ")
    policy = DQL.main_DQL(env)
    DQL.run_optimal_policy(env,policy)
    """

    print(" *** Now, Trying a Deep NN-SARSA agent:-- ")
    policy,_ = SARSA.main_SARSA(env)
    SARSA.run_optimal_policy(env,policy)
    
    print(" (Task ii):  .... ")

    print(" *** Now, Trying a Semi-Episodic SARSA agent with tile coding:-- ")
    semiEpisodicSARSA = Semi_Episodic_SARSA(env)
    semiEpisodicSARSA.Semi_Episodic_SARSA()
    semiEpisodicSARSA.run_optimal_policy()
    print(" *** **** *** **** *** **** *** ")

    env.close()