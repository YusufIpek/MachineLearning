results & values:

##########

by the way: datasets are already splited to training, validation and testing sets.


model 1: ( keras_cifar10_trained_model )
model 2: ( keras_cifar10_trained_sub_model )
model 3: ( keras_cifar10_trained_model_128_cell )
model 4: ( keras_cifar10_trained_sub_model_128_cell_64_batch )
model 5: ( keras_cifar10_trained_adam_opt )

*IMPORTANT NOTE*  I will always run 10 epochs and check the accuracy and the loss, if it looks promessing and giving a promissing values I will run 100 epochs then.

the best results are:

			* results *
	model   trial	 num_epochs		Parameter_name		previous_value		new_value		train_accuracy		test_accuracy		train_loss		test_loss		action
	1		0			100				--					--					--				0.79772			 0.7779 			0.6367213		0.70282867 		old_model
	2		0			10 				--					--					--				0.65908 		 0.6542 			0.9567476 		0.97330696 		old_model
	4		6 			100 			--		 			--					-- 				0.87464 		 0.8366 			0.37794 		0.50077 		old_model
	2		3 			10 				--		 			-- 					-- 				0.69456 		 0.6881 			0.8667949 		0.88892 		old_model
	5		8 			100				--		 			-- 					-- 				0.92114 		 0.8506 			0.238647 		0.4417384 		cur_model
	5		7 			10 				--		 			-- 					-- 				0.6884 			 0.6796 			0.90928 		0.9351026 		cur_model

	6		10			100			 	--		 			-- 					-- 			 	0.91444 		0.8524 				0.25846557 		0.4306899
	6		9			10 			 	--		 			-- 					-- 				0.69276 		 0.684 				0.88099228 		0.90294918  	under_training

our trials: 

			* done *
	2		1			10 				batch_size			32					64 				0.57726			 0.575 				1.2242891 		1.2297011 		neglect
	2		2			10 		 num_cells_initial_cnn		32 					128 			0.7216 			 0.7158 			0.79991521 		0.82979323 		apply
	2		3 			10 				batch_size 			32 					64 				0.69456 		 0.6881 			0.8667949 		0.88891986 		promissing to check
	3		4 			30 	 	num_cells_initial_cnn		32 					128 			0.7614 			 0.8011 			0.704423 		0.591712 		apply
	3		5 			100 	 num_cells_initial_cnn		32 					128 			0.76056 		 0.741 				0.813926 		0.910277 		apply early stopping
	4		6 			100 			batch_size 			32					64 				0.87464 		 0.8366 			0.37794 		0.50077 		apply
	5		7			10 				optimizer 			rmsprop				adam			0.6884 			 0.6796 			0.90928 		0.9351026 		promissing to check
	5		8 			100 			optimizer 			rmsprop 			adam 			0.92114 		 0.8506 			0.238647 		0.4417384 		apply
	5		9			10 			dropout_layer			adding dropout in_layer 0.15 	 	0.69276 		 0.684 				0.88099228 		0.90294918 		promissing to check
	5		10			100 		dropout_layer			adding dropout in_layer 0.15 	 	0.91444 		0.8524 				0.25846557 		0.4306899

			* not yet *


						10 			regularization_term		--
						10 			activation_fun_l1
						10 			arch_layer3
						10 			
			 			10 			optimizer_rate
			 			10 			batch_normalization 	--
			 			10 			Limit weight sizes 		--

train loss: 
train accuracy: 
10000/10000 [==============================] - 5s 499us/step
Test loss: 
Test accuracy: 