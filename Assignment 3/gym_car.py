import gym
# Create the gym environment
env = gym.make("MountainCar-v0")
env.seed(3333) # Set a seed for reproducability

#env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

