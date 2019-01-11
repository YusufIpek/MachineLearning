import gym

def run_RL_algorithm():
    '''
    this is just a bad function I have made, we need to call the algorithm here to choose actions acording to algorithm cariteria 
    for example use Q-learn, SARSA or TD or whatever & apply its policy to choose the right action
    '''
    '''
    TODO: add a specific RL algorithm
    TODO: make ( choose ) an action  * according to the algorithm 
    TODO: call env.step and give it that action
    TODO: update the policies and values tables, receive the rewards and returns
    TODO: continue with other requirements in the assignment
    '''
    if i % 5 ==0:
        observation, reward, done, info = env.step(action1)
    else:
        observation, reward, done, info = env.step(action2)
    return observation, reward, done, info 
    
# Create the gym environment
env = gym.make("MountainCar-v0")
#env.seed(3333) # Set a seed for reproducability

#env = gym.make('CartPole-v0')
"""
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

"""
num_episodes = 10000
# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
# env = gym.wrappers.Monitor(env, './video/', force = True)
#env.monitor.start('/tmp/video')
# monitor.start(video_callable=lambda count: count % 100 == 0) to record every 100 episodes. (count is how many episodes have completed in code)

# Number of times you want to train your agent to achieve the goal
# Equivalent to number of epochs.
env.reset()
for i in range(0, num_episodes):
    while True:
        env.render()
        #action = env.action_space.sample() # take a random action
        action1 = 0
        action2 = 2
        #print("*** chosen action: **** ",action) # actions are ( 0, 1, 2)
        
        observation, reward, done, info  = run_RL_algorithm()
        #print("returned observation: ",observation)
        #print("returned reward: ",reward)
        #print(" done or not yet: ",done)
        #print(" returned stored info about the agent: ",info)
        if done:
            print("Episode finished after {} timesteps".format(i+1))
            break
#env.close()
#env.monitor.close()        