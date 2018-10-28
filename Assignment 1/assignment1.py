import numpy as np
import random as rand
import math
from matplotlib import pyplot

N = 100
c = 6
Lambda = 5   # for regularization 
polynom_order = 9
check_case = 2

def quadratic_func(x, epsilon):
    return x**2 + x + c + epsilon

def create_epsilon_vector(variance,mean):
    epsilon_vec = np.random.normal(mean,variance,N*N)
    count, bins, ignored = pyplot.hist(epsilon_vec, 30, density=True)
    pyplot.plot(bins, 1/(variance * np.sqrt(2 * np.pi)) *np.exp( - (bins - mean)**2 / (2 * variance**2) ),linewidth=2, color='r')
    pyplot.show()
    #print(epsilon_vec)
    return epsilon_vec

def compute_epsilon(x_values):
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

def generate_data_set():
    x_values = []
    epsilon_vec = create_epsilon_vector(0,0.1)
    for i in range(0, N):
        x_values.append(i/(N/10))
    #epsilon,_ = compute_epsilon(x_values)    
    y_values = []
    for i in range(0, N):        
        y_values.append(quadratic_func(i, epsilon_vec[rand.randint(0,N*N-1)]))
        #print("Range:", i , " | Quadradit Value: " , val)
    plot_points(x_values,y_values)
    return x_values,y_values


def sigma(order, x_values):
    result = 0
    for value in x_values:
        result += value**order
    return result
   
def predict_t_value(phi,weightsVec):
    return np.matmul(phi, weightsVec)

def compute_b_matrix(matrixA, weightsVec):
    return np.matmul(matrixA, weightsVec)

def error_func(predictate_values, actual_values):
    sum_error = 0.0
    for i in range(len(actual_values)):
        predicted_error = predictate_values[i] - actual_values[i]
        sum_error += predicted_error**2
    mean_error = sum_error / float(len(actual_values))
    return math.sqrt(mean_error)

def computed_phi(x_values, order):
    big_phi_Mat = []    
    for x in x_values:
        small_phi_Vec = []
        for i in range(0,order+1):
            small_phi_Vec.append(x**i)
        big_phi_Mat.append(small_phi_Vec)
    return big_phi_Mat

def generate_general_A_matrix(m_order, x_values):
    coefficient = m_order+1
    A = []
    for i in range(0, coefficient):
        tmp = []
        for j in range(0, coefficient):
            tmp.append(sigma(i+j, x_values))
        A.append(tmp)
    return A

def generate_initial_weights_vector(polynom_order,A,b):
    return np.matmul(A,b)

def generate_initial_weights_randomly(polynom_order):
    weights_vec = []
    for i in range(0,polynom_order+1):
        weights_vec.append(rand.random())
    return weights_vec
    

def generate_b_vec(x_values,y_values,polynom_order):
    B = []
    for i in range(0,polynom_order+1):
        sum_y = 0
        for index,y in enumerate(y_values):
            sum_y += y + x_values[index]**i
        B.append(sum_y )
    return np.array(B)

def compute_new_weights(phi, predicted_values,Lambda):
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
    sum_error = 0
    weights_norm = 0
    sum_weights = 0
    for weight in weights:
        sum_weights += weight**2
    weights_norm = math.sqrt(sum_weights)
    
    for i in range(0,len(predicted_values)):
        sum_error += (predicted_values[i]-y_values[i])**2     
        
    lambda_term = (mLambda*(weights_norm**2))/2
    erms = math.sqrt((sum_error+abs(lambda_term) )/len(predicted_values))
    return erms
    
def erms_plot_k_folds(x_values, y_values, k, lambda_max):
    erms_list = []
    erms_list_trainings_set = []

    group_size = int(len(x_values)/k)
    if k > 1:
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
                
                matrixA = np.array(generate_general_A_matrix(polynom_order, tmp_x_values))
                predictedY = compute_b_matrix(matrixA, newWeightVec)
                #new_Y = np.matmul(phi,newWeightVec)
                erms_sum += root_mean_square_error(predictedY, tmp_y_values, newWeightVec, l)
                
                matrixAOfTrainSet = np.array(generate_general_A_matrix(polynom_order, training_x_values))
                predictedYOfTrainSet = compute_b_matrix(matrixAOfTrainSet, newWeightVec)
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
    lambda_set = [i for i in range(0,lambda_max)]
    plot_points_two_set(lambda_set, erms_list_trainings_set, lambda_set, erms_list)
        




x_values,y_values = generate_data_set()
#not used
#variance, mean = compute_epsilon(x_values)
#epsilon_vec = create_epsilon_vector(variance, mean)
matrixA = np.array(generate_general_A_matrix(polynom_order, x_values) )

if check_case == 1:
    #case1
    vec_B = generate_b_vec(x_values,y_values,polynom_order)
    weightVec = generate_initial_weights_vector(polynom_order,matrixA,vec_B)
else:    
    #case2
    weightVec = generate_initial_weights_randomly(polynom_order)
    vec_B = compute_b_matrix(matrixA, weightVec)

phi = np.array(computed_phi(x_values, polynom_order) )
Vec_T = predict_t_value(phi, weightVec)
newWeightVec = compute_new_weights(phi, y_values,Lambda)

new_Vec_T = predict_t_value(phi, newWeightVec)
B_temp = np.matmul((np.transpose(phi)),new_Vec_T)
new_Y = np.matmul( phi,newWeightVec )

plot_points(x_values,y_values)       
plot_points(x_values,new_Y,'red')

erms = root_mean_square_error(new_Y, y_values, newWeightVec, -Lambda)      
print("Root-Mean-Square-Error")
print(erms)

erms_plot_k_folds(x_values, y_values, 2, 10)
