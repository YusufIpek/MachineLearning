writing down some notes for each trial

trials: 

1,3 ) increasing the batch size from 32 to 64 does not give a promissing accuracy, because:
 " The batch size defines the number of samples that will be propagated through the network", as the number of memory cells in the initial input layer was only 32 cell, using 64 input doesn't make any sense, that's why it made lower accuracy.

 2,4 ) increasing the number of memory cells in the initial input layer from 32 to 128 have made a promissing progress, because:
it may depends on how it looks like in the dataset itself, I think maybe most of the pictures are with a bigger size and 32 is not enough number of cells to cover the complete image in the first step, so 128 should cover more and it fits the dataset more.

5 ) run the archticture using 128 cells for 100 epochs was not the best idea, after 40 epochs the model started to overfit the data and the accuracy decreased from 81% to around 75%, and the loss increased from 0.5 to around 0.85, that's why we thought about applying early stoping, by stop training our model before it reach overfitting state, by running only 30 epochs for our model.

P.S: We have noticied that the training accuracy was almost the same for 100 epochs, so it doesn't make sense ( should search about why this occured ).

P.S: in epoch 67 it achieved the best values : train loss: 0.7117 - train acc: 0.7704 - val loss: 0.5855 - val acc: 0.8123

6 ) in this trial, we have applied 100 epochs with 64 batch size using 128 number of memory cells of the input layer, as all of these parameters gave a promissing results earlier, and they have achieved a higher accuracy with more than 83.6% for the validation set and 87% for the training set, and achieved the lowest loss around 0.5

7,8 ) in this trial we are trying another optimizer,  the main model was using rmsprop as an optimization function and it has achieved a good results, however, we think that using Adam could be another option that have achieved better results, as Adam has the ability to adapt the learning rate iterativily.

9,10 ) Dropout is a regulization technique where you turn off part of the network's layers randomally to increase regulization and hense decrease overfitting. We use when the training set accuracy is muuch higher than the test set accuracy, Too low and you have negligible effects; too high and you underfit.
so, we have made a small dropout for the input layer, and we tried to increase the dropout ratio layer by layer, and the maximum dropout at our last hidden layer with 50% ratio, this step should help the model to achieve better accuarcy without lossing generalization.
It has showed a small progress when applying a small dropout, it has reduce the accuarcy of the training set slightly and increased the accuracy of the test set slightly, which means it has reduce the difference between the training accuracy and the test accuracy, and that encouraged us to increase the dropout for the input layer, as this will regularize the model more and increase the generality without reducing the accuracy.

11 ) trying a dropout of 25% of the input layer, showed that increasing the dropout of the input layer to 25% has reduce the accuracy and increase the loss, which means that increasing the dropout in the input layer has reduced the input features, but it does not achieve better values towards generalizing the model, so we neglect it, and consider the dropout of only 15%.

12,13 ) normal regularization on the network itself, using L2 regulaizer to regulaize all conv_layers, we have tried for small number of epochs and it was nearly the same as the results without it, so we have increased the number of epochs to give us more insights about using the L2 regulaizer, dropout for input layer was 25%

14,15,16,17 ) doing a regularization using L2 regulaizer with dropout of 0.15, has not achieve higher progress, the dropout regularization term looks more than enough to panalize the model, so we have neglect using the regularization term.

18,19 ) it is always known that data augmentation is a powerful trick that help in ..., however, during applying the data augmentation, it was clear that the data results was better than the results during the data augmentation, also trying without dropout in the input layer has shown better accuracy and promissing results.