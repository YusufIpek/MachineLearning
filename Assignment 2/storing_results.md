results & values:

##########

by the way: datasets are already splited to training, validation and testing sets.

the best results are:

for training sets: (loss: 0.6367213143348693 - accuracy: 0.79772 )
for testing sets: (loss: 0.7028286796569824 - accuracy: 0.7779 )
model 1: ( keras_cifar10_trained_model )
model 2: ( keras_cifar10_trained_sub_model )
model 3: ( keras_cifar10_trained_model_128_cell )

*IMPORTANT NOTE*  I will always run 10 epochs and check the accuracy and the loss, if it looks promessing and giving a promissing values I will run 100 epochs then.

model_num trial		num_epochs		Parameter_name		previous_value		new_value		train_accuracy		test_accuracy		train_loss		test_loss		action
1 			0			100				--					--					--				0.79772			 0.7779 			0.6367213		0.70282867 		  --
2			0			10 				--					--					--				0.65908 		 0.6542 			0.9567476 		0.97330696 		  --

* done *
2			1			10 				batch_size			32					64 				0.57726			 0.575 				1.2242891 		1.2297011 		neglect
2			2			10 		 num_cells_initial_cnn		32 					128 			0.7216 			 0.7158 			0.79991521 		0.82979323 		apply
2			3 			10 				batch_size 			32 					64 				0.69456 		 0.6881 			0.8667949 		0.88891986 		promissing to check
3			4 			30 	 	num_cells_initial_cnn		32 					128 			0.7614 			 0.8011 			0.704423 		0.591712 		apply
3			5 			100 	 num_cells_initial_cnn		32 					128 			0.76056 		0.741 				0.813926 		0.910277 		apply early stopping

			* not yet *


			6			10 				optimizer 			rmsprop				
			7			10 			regularization 			--
			8			10 			activation_fun_l1
			9			10 			arch_layer3
			10			10 			dropout_layer1
			11 			10 			optimizer_rate
			12 			10 			batch_normalization 	--


