
*PN: best parameters ( epsilon with 0.3, hidden layers = 300 )

Description:

[ for task i ]
it was about using an episodic algorithm which use tabular Q table, so we need a linear value-action function approximator.


-- Normal Q-learning


So, we have tried first the normal linear q-learning algorithm but after many tries ( different learning rates, different exploration & gready threshold epsilon, different discount factors ) the agent could not learn the appropriate policy, and the main reason of this issue was the rewarding system, giving a -1 at each iteration was not a great indecation about the more you spend iterations the more punish you get, and theoritically this is a good point, however, because the agent never receive a positive reward till it reachs the goal, the agent was not able to figure out that it could be close to the goal but never explore it.
and this raise the conflict between do more exploration to search about the goal or reduce the number of iterations to reduce the negative reward.

0/3000 successful episodes ( epsilon  = 0.6 )
0/3000 successful episodes ( epsilon  = 0.3 )
0/3000 successful episodes ( epsilon  = 0.2 )
0/3000 successful episodes ( epsilon  = 0.1 )


-- Deep Q-learning

after that, we dicided to check a better approach by using deep q-learning using a nueral network with one linear layer.

The portion inside the brackets becomes the loss function for our neural network where Q(st,at) is the output of our network and rt + γ max Q(st+1,at+1) is the target Q value as well as the label for our neural net turning the problem into a supervised learning problem solvable using gradient descent where α is our learning rate.


one of the most important parameters that needed to be tune is the exploration epsilon gready parameter, this paremeter is the exploration versus explotation parameter, choosing the epsilon high will make the agent act greedy with less exploration, in such a problem exploration is critical as the agent get a positive reward only when it reachs the goal.

All the following trials were made with one hidden layer with 200 units:

0/3000 successful episodes, no successful episode ( epsilon  = 0.6 )
37/3000 successful episodes, first success at 1340 ( epsilon  = 0.4 )
626/3000 successful episodes, first success at 1020 ( epsilon  = 0.3 )
426/3000 successful episodes, first success at 909 ( epsilon  = 0.2 )
397/3000 successful episodes, first success at 850 ( epsilon  = 0.1 )

Trying one hidden layer with 300 units has achieved better results:

729/3000 successful episodes, first success at 846 ( epsilon  = 0.3 )


[ Then for task ii ]
wew are required to use a non-linear value-action function approximation using the continues value-action function approximator using the polynomial feature vector.

