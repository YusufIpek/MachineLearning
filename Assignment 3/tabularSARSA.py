import numpy as np
import gym

def main_QL(env,episodes = 20000):
    n_states = 250
    np.random.seed(3333)
    q_table = np.zeros((n_states, n_states, env.action_space.n))
    q_table,successful_tries,first_succeeded_episode = train(env,q_table,n_states = n_states,max_episodes = episodes,render=False)
    print(" succeeded episodes: ",successful_tries)
    print(" The first episode that reached the solution is: ",first_succeeded_episode)
    solution_policy = np.argmax(q_table, axis=2)
    print("Solution policy")
    print(q_table)

    # Animate it
    solution_policy_scores = [run(env,n_states = n_states,render=False, policy=solution_policy) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    for _ in range(5):
        # in case of solving the problem, to just provide the solution directly
        run(env,render=True, policy=solution_policy,n_states = n_states)

def train(env,q_table,n_states = 50,max_episodes = 20000,discount_factor = 0.99,initial_lr = 0.1,epsilon = 0.3 ,min_lr = 0.001,max_iterations = 10000 ,render=False):
    '''
    tabular-SARSA algo:
    - Initialize parameters
    - Initialize Policy model and optimizers
    - for each episode
        - Initialize state S
        - choose action A using the policy  
        - for each step in the episode
            - take a step with action A & get the reward R and next state S'
            - choose action A' for the next state S' using the policy
            - update the policy of the Q table 
            - update the action A = A' & the state S = S'
    '''
    first_succeeded_episode = -1
    successful_tries = 0
    for ep in range(max_episodes):
        S = env.reset()
        total_reward = 0
        ## updated_lr: learning rate is decreased at each step
        updated_lr = max(min_lr, initial_lr * (0.85 ** (ep//100)))
        #state = obs_to_state(env,obs,n_states)
        if np.random.uniform(0, 1) < epsilon: # act non-greedy or state-action have no value # exploration constant 
            A = np.random.choice(env.action_space.n)
        else:
            A = q_table[S]
        for iter in range(max_iterations):
            if np.random.uniform(0, 1) < epsilon: # act non-greedy or state-action have no value # exploration constant 
                A = np.random.choice(env.action_space.n)
            if render:
                env.render()
            
            S_1, reward, done, info = env.step(A)
            total_reward += reward
            # update q table
            A_1 = q_table[S_1]
            # q_table[S] = q_table[S] + updated_lr * (reward + discount_factor * q_table[S_1] - q_table[S])
            q_table[S + (A,)] = q_table[S + (A,)] + updated_lr * (reward + discount_factor *  q_table[S_1 + (A_1,)] - q_table[S + (A, )])
            if done:
                if S_1[0] >= 0.5:
                    # Store episode number if it is the first
                    if successful_tries == 0:
                        first_succeeded_episode = ep
                    successful_tries +=1

                if iter + 1 != 200 :
                    print("Episode finished after {} timesteps".format(iter + 1))
                break
            S = S_1
            A = A_1
        if ep % 500 == 0:
            print('Iteration #{} -- Total reward = {}.'.format(ep+1, total_reward))
    return q_table,successful_tries,first_succeeded_episode

def run(env,render=True, policy=None,discount_factor = 0.99,max_iterations = 10000,n_states = 50):

    S = env.reset()
    total_reward = 0
    step_idx = 0
    if policy is None:
        A = env.action_space.sample()
    else:
        # just applying the learned policy
        A = policy[S]
    for iter in range(max_iterations):
        if render:
            env.render()
        S_1, reward, done, info = env.step(A)
        A_1 = policy[S_1]

        if S_1[0] >= 0.5:
                # Store episode number if it is the first
                print(" Successful try in testing phase, Car reached the goal.")

        total_reward += discount_factor ** step_idx * reward
        step_idx += 1
        if done:
            if iter + 1 != 200:
                print("Episode finished after {} timesteps".format(iter + 1))
            break
        S = S_1
        A = A_1
    return total_reward

def obs_to_state(env,obs,n_states):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return [a, b]



if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(3333)
    main_QL(env)