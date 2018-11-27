writing down some notes for each trial

trials: 

1 ) increasing the batch size from 32 to 64 does not give a promissing accuracy, because:
 " The batch size defines the number of samples that will be propagated through the network", as the number of memory cells in the initial input layer was only 32 cell, using 64 input doesn't make any sense, that's why it made lower accuracy.

 2 ) increasing the number of memory cells in the initial input layer from 32 to 128 have made a promissing progress, because:
it may depends on how it looks like in the dataset itself, I think maybe most of the pictures are with a bigger size and 32 is not enough number of cells to cover the complete image in the first step, so 128 should cover more and it fits the dataset more.