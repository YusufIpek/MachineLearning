import numpy as np
import random as rand
import math
from matplotlib import pyplot

N = 10
c = 6
Lambda = 3   # for regularization 
polynom_order = 4

def quadratic_func(x, epsilon):
    return x**2 + x + c + epsilon

def create_epsilon_vector():
    mean = 0
    sigma = 0.1
    samples = N*N
    epsilon_vec = np.random.normal(mean,sigma,samples)
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
    epsilon_vec = create_epsilon_vector()
    for i in range(0, N):
        x_values.append(i)
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

def generate_general_A_matrix(m_order, x_values):
    coefficient = m_order+1
    A = []
    for i in range(0, coefficient):
        tmp = []
        for j in range(0, coefficient):
            tmp.append(sigma(i+j, x_values))
        A.append(tmp)
    return A
   
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
    # for each x value in vector of x_values we loop and make phi vector for each x
    
    big_phi_Mat = []
    
    for x in x_values:
        small_phi_Vec = []
        for i in range(0,order+1):
            small_phi_Vec.append(x**i)
        big_phi_Mat.append(small_phi_Vec)
    return big_phi_Mat


def generate_initial_weights_vector(polynom_order):
    weights_vec = []
    for i in range(0,polynom_order+1):
        weights_vec.append(rand.random())
    return weights_vec


def compute_new_weights(phi, predicted_values,Lambda):
    before_lambda = np.matmul(np.transpose(phi),phi)
    shape_tuple = before_lambda.shape
    regularization_matrix = Lambda*(np.identity(shape_tuple[0]))
    tmp = np.linalg.inv(before_lambda + regularization_matrix)
    tmp = np.matmul(tmp,np.transpose(phi))
    return np.matmul( tmp,predicted_values )

def plot_points(data_x,data_y,clr = 'blue'): 
    pyplot.scatter(data_x,data_y)
    pyplot.scatter(data_x,data_y,color=clr)
    pyplot.show()
        

x_values,y_values = generate_data_set()
variance, mean = compute_epsilon(x_values)    
#epsilon_vec = create_epsilon_vector(variance, mean)
#print(data_set)
matrixA = np.array(generate_general_A_matrix(polynom_order, x_values) )
#print("A matrix:")
#print(matrixA)
  
weightVec = generate_initial_weights_vector(polynom_order)    
print("Initial weights:")
print(weightVec)

bVec = compute_b_matrix(matrixA, weightVec)
#print("B Vector:")
#print(bVec)

phi = np.array(computed_phi(x_values, polynom_order) )
#print("Phi:")#design Matrix
#print(phi)

Vec_T = predict_t_value(phi, weightVec)
#print("T Vector")
#print(Vec_T)

B_temp = np.matmul((np.transpose(phi)),Vec_T)
#print("B tmppp")
#print(B_temp)


newWeightVec = compute_new_weights(phi, Vec_T,Lambda)
print("New Weight")
print(newWeightVec)

new_Y = np.matmul( phi,newWeightVec )
print("New Y values")
#print(new_Y) 
plot_points(x_values,new_Y,'red')
#plot_points(x_values,y_values)       
