#Tasks for RL assignment

1- read and study RL lectures.

2- install gym open source library and set the environment 

'pip install gym'

if you got error with running on windows, please check this link [https://github.com/openai/gym/issues/868]

3- try to solve the mountain car task using linear value function (v fun)

	- using TD & V(s,a)
	- action-value function Q(s,a)
	- polynomial feature vector
	- try multiple runs 

4- create reports and blot graphs for observations and results

we have tried first the normal q-learning but after many tries ( different learning rates, different exploration & gready threshold epsilon, different discount factors ) the agent could not learn the appropriate policy, and the main reason of this issue was the rewarding system, giving a -1 at each iteration was not a great indecation about the more you spend iterations the more punish you get, and theoritically this is a good point, however, because the agent never receive a positive reward till it reachs the goal, the agent was not able to figure out that it could be close to the goal but never explore it.
and this raise the conflict between do more exploration to search about the goal or reduce the number of iterations to reduce the negative reward.

after mthat, we dicided to check a better approach by using deep q-learning
The portion inside the brackets becomes the loss function for our neural network where Q(st,at) is the output of our network and rt + γ max Q(st+1,at+1) is the target Q value as well as the label for our neural net turning the problem into a supervised learning problem solvable using gradient descent where α is our learning rate.

5- prepare report



 [ some useful links ]

this repo contains useful algo, and tutorials

 https://github.com/vmayoral/basic_reinforcement_learning

- it also has a tutorial about the openAI gym library we work with

https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial3/README.md

- another tutorial about using Q-learning to solve a problem in gym library 

https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4/README.md
