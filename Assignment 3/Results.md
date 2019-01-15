
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
[ Then for task ii ]
wew are required to use a linear value-action function approximation using the continues value-action function approximator using the polynomial feature vector.

