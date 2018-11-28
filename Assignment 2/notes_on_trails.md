writing down some notes for each trial

trials: 

1,3 ) increasing the batch size from 32 to 64 does not give a promissing accuracy, because:
 " The batch size defines the number of samples that will be propagated through the network", as the number of memory cells in the initial input layer was only 32 cell, using 64 input doesn't make any sense, that's why it made lower accuracy.

 2,4 ) increasing the number of memory cells in the initial input layer from 32 to 128 have made a promissing progress, because:
it may depends on how it looks like in the dataset itself, I think maybe most of the pictures are with a bigger size and 32 is not enough number of cells to cover the complete image in the first step, so 128 should cover more and it fits the dataset more.

5 ) run the atchticture using 128 cells for 100 epochs was not the best idea, after 40 epochs the model started to overfit the data and the accuracy decreased from 81% to around 75%, and the loss increased from 0.5 to around 0.85, that's why we thought about applying early stoping, by stop training our model before it reach overfitting state, by running only 30 epochs for our model.

P.S: We have noticied that the training accuracy was almost the same for 100 epochs, so it doesn't make sense ( should search about why this occured ).

P.S: in epoch 67 it achieved the best values : train loss: 0.7117 - train acc: 0.7704 - val loss: 0.5855 - val acc: 0.8123

6 ) in this trial, we have applied 100 epochs with 64 batch size using 128 number of memory cells of the input layer, as all of these parameters gave a promissing results earlier, and they have achieved a higher accuracy with more than 83.6% for the validation set and 87% for the training set, and achieved the lowest loss around 0.5

7 ) in this trial we are trying another optimizer,  the main model was using rmsprop as an optimization function and it has achieved a good results, however, we think that using Adam could be another option that have achieved better results, as Adam has the ability to adapt the learning rate iterativily.

8 ) 

