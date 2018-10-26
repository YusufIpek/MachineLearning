import numpy as np
import random as rand
import math
from matplotlib import pyplot

N = 100
c = 6
Lambda = 18   # for regularization 
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


def root_mean_square_error(predicted_values, y_values, weights):
    sum_error = 0
    weights_norm = 0
    sum_weights = 0
    for weight in weights:
        sum_weights += weight**2
    weights_norm = math.sqrt(sum_weights)
    
    for i in range(0,len(predicted_values)):
        sum_error += (predicted_values[i]-y_values[i])**2 + (Lambda*(weights_norm**2))/2
    erms = math.sqrt((2*sum_error)/len(predicted_values))
    return erms
    
        

x_values,y_values = generate_data_set()
variance, mean = compute_epsilon(x_values)    
epsilon_vec = create_epsilon_vector(variance, mean)
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

erms = root_mean_square_error(new_Y, y_values, newWeightVec)      
print("Root-Mean-Square-Error")
print(erms)

