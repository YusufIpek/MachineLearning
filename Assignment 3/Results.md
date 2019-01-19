
*PN: best parameters ( epsilon with 0.2, hidden layers = 500 )

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

               =======================================================================================================================================
It totally make sense that tabular methedologies did not work, because tabular methedology works fine with small problems with small environment, although our environment does not look so complex but it is not small and tabular methods doesn't achieve good results with it. thinking deeply about our problem space, we have only three actions to get to observations about the state, the position and the velocity of the agent, and that's why combining the velocity and the position together generate a large environment that needs a function approximation methedologies paramaterized by some weight vector.
In these methedologies, the RL algorithms updates the parameters ( weight vector ) during the training phase.

-- Deep Q-learning

That's why, we dicided to check a better approach thinking about supervised machine learning by using a nueral network with one linear layer to build a deep q-learning agent.

The portion inside the brackets becomes the loss function for our neural network where action-value function is the output of our network. and rt + gamma max Q(st+1,at+1) is the target value as well as the label for our neural network turning the problem into a supervised learning problem solvable using gradient descent.


During this experiment we have some parameters to tune, one of the most important parameters that needed to be tuned is the exploration epsilon gready parameter, this paremeter is the exploration versus explotation parameter, choosing the epsilon high will make the agent act greedy with less exploration, in such a problem exploration is critical as the agent get a positive reward only when it reachs the goal.

All the following trials were made with one hidden layer with 200 units:

0/3000 successful episodes, no successful episode ( epsilon  = 0.6 )
37/3000 successful episodes, first success at 1340 ( epsilon  = 0.4 )
626/3000 successful episodes, first success at 1020 ( epsilon  = 0.3 )
426/3000 successful episodes, first success at 909 ( epsilon  = 0.2 )
397/3000 successful episodes, first success at 850 ( epsilon  = 0.1 )

Trying different units number for the hidden layer has achieved better results:

729/3000 successful episodes, first success at 846 ( epsilon  = 0.3, hidden layer units = 300 )
659/3000 successful episodes, first success at 728 ( epsilon  = 0.2, hidden layer units = 300 )

834/3000 successful episodes, first success at 720 ( epsilon  = 0.3, hidden layer units = 500 )
670/3000 successful episodes, first success at 584 ( epsilon  = 0.2, hidden layer units = 500 )

               =======================================================================================================================================

-- SARSA
after trying with DeepQL, we have used SARSA semi-episodic algorithm, and it has shown much better results than Deep Q-Learning algorithm:

226/3000 successful episodes, first success at 768 ( epsilon  = 0.1, hidden layer units = 300 )
624/3000 successful episodes, first success at 794 ( epsilon  = 0.3, hidden layer units = 300 )

876/3000 successful episodes, first success at 621 ( epsilon  = 0.3, hidden layer units = 500 )

615/3000 successful episodes, first success at 663 ( constant epsilon  = 0.2, hidden layer units = 500 )
923/3000 successful episodes, first success at 663 ( decreasing epsilon  = 0.2, hidden layer units = 500 )
93/3000 successful episodes, first success at 663 ( increasing epsilon  = 0.2, hidden layer units = 500 )

it is important to notice that increasing the number of hidden layer units have made a better performance increasing the number of total successful episodes, the more units in the hidden layer the more successful episodes we got.

               =======================================================================================================================================

*on the testing phase:
=======================

QL did not succeeed
DeepQL succeeded in 5 out 10 in 0.3 epsilon
DeepQL succeeded in 5 out 10 in 0.3 epsilon

SARSA succeeded in 4 out 10 in 300 hidden layers unit

SARSA succeeded in 6 out 10 in 500 hidden layers unit ( decreasing epsilon 0.2 )
SARSA succeeded in 7 out 10 in 500 hidden layers unit ( constant epsilon 0.2 )
SARSA succeeded in 8 out 10 in 500 hidden layers unit ( increasing epsilon 0.2 )

we also tried to adjust the epsilon ( the greedy parameter ), by slightly reducing it after each successful episode ( to act less greedy ) and surprisingly this increased the number of successful episodes in the training phase but reduce slightly the number of successful episodes in the testing phase, on the other hand, increasing the epsilon after every successful episode made the agent more greedy and reduce the exploration, hence, the number of successful episodes in the training phase have decreased deramatically, but in the testing phase, it has increased much more.

                ==========================================                      =========================================


we then tried the semi-episodic algorithm to solve the problem without using any neural network archticture, and surprisingly, it has shown a very good progress, although it may take much time for training, but overall it does not need the same number of episodes to find the goal for the first time, and after few episodes it can convarge.
using semi-episodic algorithm has achieved a real convargance if we compared with earlier experiments. 


[ Then for task ii ]
we are required to use a linear value-action function approximation using the continues value-action function approximator using the polynomial feature vector.

we have used a ready code doing tiling by Richard S. Sutton, the idea behind tiling is just dividing the space into number of partitions, we have chosen 16 partitions as recommended from the lectures, each partition is a tiling, and element in the tiling is a tile, we are creating the features of each partition by combining the position and the velocity of the object to have a feature represents these tiles in its tiling partition.

 One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/tiles/tiles3.html

after trying some iterations, we found that our space is small to do a tiling for 16 partition, so we have tried 8 partitions and it has shown better results in finding a feature vector to represent it.