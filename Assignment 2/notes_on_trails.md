writing down some notes for each trial

trials: 

1 ) increasing the batch size from 32 to 64 does not give a promissing accuracy, because:
 " The batch size defines the number of samples that will be propagated through the network", as the number of memory cells in the initial input layer was only 32 cell, using 64 input doesn't make any sense, that's why it made lower accuracy.

 2 ) increasing the number of memory cells in the initial input layer from 32 to 128 have made a promissing progress, because:
