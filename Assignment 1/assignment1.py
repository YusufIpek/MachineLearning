import numpy as np
import random as rand
import math
import sys
from matplotlib import pyplot

def generate_data_set(N):
    """
        this function is to generate the points (x,y)
        N: number of points to generate
    """
    x_values = []
    epsilon_vec = create_epsilon_vector(0,0.1,N)
    for i in range(0, N):
        x_values.append(i/(N/10))  
    y_values = []
    for i in range(0, N):        
        y_values.append(quadratic_func(i, epsilon_vec[rand.randint(0,N*N-1)]))
        #print("Range:", i , " | Quadradit Value: " , val)
    #plot_points(x_values,y_values)
    return x_values,y_values

def create_epsilon_vector(variance,mean,N):
    
    """
        using a constant variance = 0 and a mean = 0.1 giving a good normal distributed points
    """
    epsilon_vec = np.random.normal(mean,variance,N*N)
    # for ploting the normal distribution
    #count, bins, ignored = pyplot.hist(epsilon_vec, 30, density=True)
    #pyplot.plot(bins, 1/(variance * np.sqrt(2 * np.pi)) *np.exp( - (bins - mean)**2 / (2 * variance**2) ),linewidth=2, color='r')
    #pyplot.show()
    #print(epsilon_vec)
    return epsilon_vec

def quadratic_func(x, epsilon):
    """
        non-linear quadratic function to generate the y value for any received x point ( f(x) function )
    """
    return x**2 + x + 6 + epsilon     # ( x^2 + x + const )

def compute_epsilon(x_values):
    """
        this function is to compute the epsilon by calculate the mean and the variance of the x values
    """
    if len(x_values) == 0:
        return -1
    sum_of_x = 0
    for x in x_values:
       sum_of_x += x
    mean = sum_of_x/len(x_values)
    
    normalized_sum = 0
    for x in x_values:
        normalized_sum += (x-mean)**2
    return (normalized_sum/len(x_values)),mean    

def predict_t_value(phi,weightsVec):
    """
        this function receive the weights and phi to calculate the y using the following equ: Tn = Phi .* W
    """
    return np.matmul(phi, weightsVec)

def compute_b_matrix(matrixA, weightsVec):
    """
        this function compute the b matrix using the follwoing equ: b = A .* W
    """
    return np.matmul(matrixA, weightsVec)

def computed_phi(x_values, order):
    """
        this function to compute the PHI ( design matrix )
    """
    big_phi_Mat = []    
    for x in x_values:
        small_phi_Vec = []
        for i in range(0,order+1):
            small_phi_Vec.append(x**i)
        big_phi_Mat.append(small_phi_Vec)
    return big_phi_Mat

def predict_values(weights, x_values):
    """
        this function used to predict the y values for the error functions
    """
    result = []
    for x in x_values:
        tmp = 0
        for i in range(0,len(weights)):
            if i == 0:
                tmp += weights[i]
            else:
                tmp += (x**i) * weights[i]
        result.append(tmp)
    return result
            
def sigma(order, x_values):
    """
        to perform the summation over sigma used to generate the A matrix
    """
    result = 0
    for value in x_values:
        result += value**order
    return result
   
def generate_general_A_matrix(m_order, x_values):
    """
        this function to compute the A matrix from the x values
    """
    coefficient = m_order+1
    A = []
    for i in range(0, coefficient):
        tmp = []
        for j in range(0, coefficient):
            tmp.append(sigma(i+j, x_values))
        A.append(tmp)
    return A

def generate_initial_weights_randomly(polynom_order):
    """
        to calculate the weights randomly
    """
    weights_vec = []
    for i in range(0,polynom_order+1):
        weights_vec.append(rand.random())
    return weights_vec

def compute_new_weights(phi, predicted_values,Lambda):
    """
        calculate the new Weights using the pseudo-inverse 
    """
    before_lambda = np.matmul(np.transpose(phi),phi)
    shape_tuple = before_lambda.shape
    regularization_matrix = Lambda*(np.identity(shape_tuple[0]))
    tmp = np.linalg.inv(before_lambda + regularization_matrix)
    tmp = np.matmul(tmp,np.transpose(phi))
    return np.matmul( tmp,predicted_values )

def plot_points(data_x,data_y,clr = 'blue'):
    pyplot.scatter(data_x,data_y,color=clr)
    pyplot.show()

def plot_points_two_set(data_training_x, data_training_y, data_test_x, data_test_y):
    training = pyplot.scatter(data_training_x, data_training_y, color='blue')
    test = pyplot.scatter(data_test_x, data_test_y, color='red')
    pyplot.legend((training, test),('Traning', 'Test'))
    pyplot.show()


def root_mean_square_error(predicted_values, y_values, weights, mLambda):
    """
        this function to calculate the Error function.
    """
    sum_error = 0
    weights_norm = 0
    sum_weights = 0
    for weight in weights:
        sum_weights += weight**2
    weights_norm = math.sqrt(sum_weights)
    
    for i in range(0,len(predicted_values)):
        sum_error += (predicted_values[i]-y_values[i])**2     
        
    lambda_term = (mLambda*(weights_norm**2))/2
    Erms = math.sqrt((sum_error+abs(lambda_term) )/len(predicted_values))
    return Erms
    
def erms_plot_k_folds(x_values, y_values, k, lambda_max, polynom_order,new_Y):
    erms_list = []
    erms_list_trainings_set = []

    group_size = int(len(x_values)/k)
    
    if group_size == 1:
        sys.exit('[erms_plot_k_folds] K should be choosen so that the training set should at least contain 2 elements!')
    
    if k > 1: # cross validation can be done
        for l in range(0,lambda_max):
            erms_sum = 0
            erms_sum_training_set = 0
            for i in range(0,k-1):
                if i == k-1:
                    #last iteration should get the rest
                    training_x_values = x_values[i*group_size:len(x_values)]
                    training_y_values = y_values[i*group_size:len(y_values)]
                else: 
                    training_x_values = x_values[i*group_size:(i*group_size)+group_size]
                    training_y_values = y_values[i*group_size:(i*group_size)+group_size]
                
                #generate phi with new training x values
                phi = np.array(computed_phi(training_x_values, polynom_order))
                #compute new weight with new training y values
                newWeightVec = compute_new_weights(phi, training_y_values,l)
                
                tmp_x_values = []
                tmp_y_values = []
                if i == k-1:    
                    tmp_x_values = x_values[0:i*group_size]
                    tmp_y_values = y_values[0:i*group_size]
                elif i == 0:
                    tmp_x_values = x_values[group_size:len(x_values)]
                    tmp_y_values = y_values[group_size:len(y_values)]
                else:
                    tmp_x_values = x_values[0:i*group_size]
                    tmp_x_values.extend(x_values[i*group_size + group_size:len(x_values)])
                    
                    tmp_y_values = y_values[0:i*group_size]
                    tmp_y_values.extend(y_values[i*group_size + group_size:len(y_values)])
                
                #matrixA = np.array(generate_general_A_matrix(polynom_order, tmp_x_values))
                #trainingsPhi = np.matmul(np.array(computed_phi(tmp_x_values, polynom_order)), newWeightVec)
                #predictedY = compute_b_matrix(matrixA, newWeightVec)
                predictedY = predict_values(newWeightVec, tmp_x_values)
                #new_Y = np.matmul(phi,newWeightVec)
                erms_sum += root_mean_square_error(predictedY, tmp_y_values, newWeightVec, l)
                
                #matrixAOfTrainSet = np.array(generate_general_A_matrix(polynom_order, training_x_values))
                #testPhi = np.matmul(np.array(computed_phi(training_x_values, polynom_order)), newWeightVec)
                #predictedYOfTrainSet = compute_b_matrix(matrixAOfTrainSet, newWeightVec)
                predictedYOfTrainSet = predict_values(newWeightVec, training_x_values)
                erms_sum_training_set += root_mean_square_error(predictedYOfTrainSet, training_y_values, newWeightVec, l)
            mean_erms = erms_sum/(k-1)
            erms_list.append(mean_erms)
            mean_erms_of_training_set = erms_sum_training_set/(k-1)
            erms_list_trainings_set.append(mean_erms_of_training_set)
    else:
        
        for l in range(0,lambda_max):
            phi = np.array(computed_phi(x_values, polynom_order))
            newWeightVec = compute_new_weights(phi, y_values,l)
            erms_list.append(root_mean_square_error(new_Y, y_values, newWeightVec, l))
        
    #plot_points([i for i in range(0,lambda_max)], erms_list)
    lambda_set = [-i for i in range(0,lambda_max)]
    plot_points_two_set(list(reversed(lambda_set)), list(reversed(erms_list_trainings_set)), list(reversed(lambda_set)), list(reversed(erms_list)))
        
def plot_with_and_without_regularization(x_values, y_values, lambda_list,polynom_order):
    phi = np.array(computed_phi(x_values, polynom_order) )
    result = []
    for mLambda in lambda_list:
        newWeightVec = compute_new_weights(phi, y_values,mLambda)
        bVec = np.matmul(phi, newWeightVec)
        result.append(bVec)
    
    colors = ['red', 'blue', 'green', 'yellow']
    counter = 0
    plots = []
    for res in result:
        plots.append(pyplot.scatter(x_values, list(map(lambda x: x/100,res)), color=colors[counter]))
        counter += 1
    pyplot.legend((plots[0],plots[1],plots[2],plots[3]),
                  ('lambda:'+str(lambda_list[0]),
                   'lambda:'+str(lambda_list[1]),
                   'lambda:'+str(lambda_list[2]),
                   'lambda:'+str(lambda_list[3])))
    pyplot.show()        


def linear_regrssion_model(NUM_Points = 100, Lambda = 5 , polynom_order = 8, k_folds = 6):
    """
        main function to perform the linear regression model.
        
        NUM_Points: number of generated points
        Lambda: lambda value  for regularization 
        polynom_order: the M-th polynomial order value
    """

    # generate data-sets of (x,y) points
    x_values,y_values = generate_data_set(NUM_Points)
    
    # generate the known matrices for A.*W = b 
    matrix_A = np.array(generate_general_A_matrix(polynom_order, x_values) )
    weightVec = generate_initial_weights_randomly(polynom_order)
    vec_B = compute_b_matrix(matrix_A, weightVec)

    # Design Matrix PHI
    phi = np.array(computed_phi(x_values, polynom_order) )
    
    # calculate the new Weights using the pseudo-inverse 
    newWeightVec = compute_new_weights(phi, y_values,Lambda)
    print(newWeightVec)
    
    # calculate the new generated Y_values for validating solution
    new_Y = np.matmul( phi,newWeightVec )    
    #plot_points(x_values,y_values)       
    #plot_points(x_values,new_Y,'red')
    
    erms = root_mean_square_error(new_Y, y_values, newWeightVec, Lambda)      
    
    erms_plot_k_folds(x_values, y_values, k_folds, 10,polynom_order,new_Y)    
    plot_with_and_without_regularization(x_values, y_values, [0,-18,-36,-72],polynom_order)

if __name__ == "__main__":
    linear_regrssion_model(NUM_Points = 100, Lambda = 5 , polynom_order = 8, k_folds = 10)
