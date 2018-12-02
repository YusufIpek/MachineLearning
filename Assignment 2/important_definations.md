Tuning a model in deep learning is always a tricky part, we have a lot of hyper-parameters that we can tune, and our model will extremely depend on them, and they are totally depend on the type of the problem the model tries to solve, most of the time following the normal procedures will get a good model, but achieving a very good model without lossing genarlization is a difficult point that need a lot of understanding of your problem, your dataset, and your archticture frame. all these points should drive us to tune our model with the suitable hyper-parameters.

important definations


Term:						Description

#one epoch# 				One forward pass and one backward pass of all the training examples

#batch size#				The number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.

#early stopping#			Stop training the model before reaching an overfitting state.

#Dropout#					Dropout is a regulization technique where you turn off part of the network's layers randomally to increase regulization and hense decrease overfitting. 
							We use when the training set accuracy is muuch higher than the test set accuracy.

#Max Pooling#	  			The maximum output in a rectangular neighbourhood. It is used to make the network more flexible to slight changes and decrease the network computationl expenses
							by extracting the group of pixels that are highly contributing to each feature in the feature maps in the layer.

#Convolutional layers#  	The convolutional layer is responsible for the convolutional operation in which feature maps identifies features in the images. 

#Dense layers#	 			The dense layer is a fully connected layer that give us the output vector of the Network.

#Adam optimizer#			adaptive moment estimation, Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network 								weights iterative based in training data, by adapting it is values according to the current state of the weights.

#Rmsprop optimizer#			utilizes the magnitude of recent gradients to normalize the gradients. by uses a moving average of the root mean squared gradients. That has an effect of 									balancing the step size — decrease the step for large gradient to avoid exploding, and increase the step for small gradient to avoid vanishing.

#Regularization#			regularization is a way of limit the weights values to getting higher values that produce overfitting, regulaization is done by panalize high weights of the 								cooefecients, by adding a penality term.
							regularization has 2 main types of regularization L1 & L2.


batch_size					
					 		determines the number of samples in each mini batch. Its maximum is the number of all samples, which makes gradient descent accurate, the loss will decrease towards the minimum if the learning rate is small enough, but iterations are slower. Its minimum is 1, resulting in stochastic gradient descent: Fast but the direction of the gradient step is based only on one example, the loss may jump around. batch_size allows to adjust between the two extremes: accurate gradient direction and fast iteration. Also, the maximum value for batch_size may be limited if your model + data set does not fit into the available (GPU) memory.
steps_per_epoch 
							the number of batch iterations before a training epoch is considered finished. If you have a training set of fixed size you can ignore it but it may be useful if you have a huge data set or if you are generating random data augmentations on the fly, i.e. if your training set has a (generated) infinite size. If you have the time to go through your whole training data set I recommend to skip this parameter.


FIT_GENERATOR
One great advantage about fit_generator() besides saving memory is user can integrate random augmentation inside the generator, so it will always provide model with new data to train on the fly.

TIPS

For Dropout:

Use small dropouts of 20–50%, with 20% recommended for inputs. Too low and you have negligible effects; too high and you underfit.
Use dropout on the input layer as well as hidden layers. This has been proven to improve deep learning performance.


For network archticture:

Use a larger network. You are likely to get better performance when dropout is used on a larger network, giving the model more of an opportunity to learn independent representations.
The first hidden layers of a neural network tend to capture universal and interpretable features, like shapes, curves, or interactions that are very often relevant across domains. We should often leave these alone, and focus on optimizing the meta² latent level further back. This may mean adding hidden layers so we don’t rush the process!

for weights sizes:

Limit weight sizes of the parameters will help to generalize the model.	 	
We can limit the max norm (absolute value) of the weights for certain layers in order to generalize our model
	kernel_initializer='normal', kernel_constraint=maxnorm(5)
Constrain your weights! A big learning rate can result in exploding gradients. Imposing a constraint on network weight — such as max-norm regularization with a size of 5 — has been shown to improve results.

for flatten layers:

we add a flatten layer that takes the output of the CNN and flattens it and passes it as an input to the Dense Layers which passes it to the output layer. 

for optimization:

using Adam or rmsprop, as they have a very good flixability, ..... ( describe them more ).



very useful

Input Layer : 

All the input layer does is read the image. So, there are no parameters learn in here.

Convolutional Layer : 

Consider a convolutional layer which takes “l” feature maps as the input and has “k” feature maps as output. The filter size is “n*m”.
Here the input has l=32 feature maps as inputs, k=64 feature maps as outputs and filter size is n=3 and m=3. It is important to understand, that we don’t simply have a 3*3 filter, but actually, we have 3*3*32 filter, as our input has 32 dimensions. And as an output from first conv layer, we learn 64 different 3*3*32 filters which total weights is “n*m*k*l”. Then there is a term called bias for each feature map. So, the total number of parameters are “(n*m*l+1)*k”.

number of filters

we have always a 3*3 kernel ,right ?
that kernel is one of the weights that we try to learn the new information that we can say the first parameters is the number of the filters we have so we have around 28*28 hidden layer, so this hidden layer had 784(28*28) cell so we have already multiplied it with our filter 784 the point here is instead of using one filter we say we will have 32 filters or 64 or 128 according to our complixety of our problem and we want to learn the accurate weights of these filters

so how to pick a good number of filters, it must be higher than the number of cells in our kernel 
if we use 5*5*1 then it should be higher than 25 
if we use 5*5*3 then it should be higher than 75 
and so on and I am sure it should be a high number because we have many different features in the images ( such as the edges , the straight lines, the curves, corners, different colors and so on ) but why it should not be too much higher, because imagine we have 784 filter for each image it will not describe main features like lines and edges , but it will then describe wrongly the feature of the image itself and this is overfitting
-- more illosturation:
When you are using the network to make a prediction, these filters are applied at each layer of the network. That is, a discrete convolution is performed for each filter on each input image, and the results of these convolutions are fed to the next layer of convolutions (or fully connected layer, or whatever else you might have).

During training, the values in the filters are optimised with backpropogation with respect to a loss function. For classification tasks such as recognising digits, usually the cross entropy loss is used. Here's a visualisation of some filters learned in the first layer (top) and the filters learned in the second layer (bottom) of a convolutional network:
As you can see, the first layer filters basically all act as simple edge detectors, while the second layer filters are more complex. As you go deeper into a network, the filters are able to detect more complex shapes. It gets a little tricky to visualise though, as these filters act on images that have been convolved many times already, and probably don't look much like the original natural image.