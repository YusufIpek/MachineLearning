import numpy as np
import random as rand
import math

N = 10
c = 6

def quadratic_func(x):
    return x**2 + x + c


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
    
    return normalized_sum/len(x_values)
    
        
    

def generate_data_set():
    tmp_data_set = []
    
    for i in range(0, N):
        tmp_data_set.append(i)
    epsilon = compute_epsilon(tmp_data_set)
    
    data_set = []
    for i in range(0, N):        
        val = quadratic_func(i)
        #print("Range:", i , " | Quadradit Value: " , val)
        data_set.append(val+epsilon)
    
    return data_set


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


def generate_weights_vector(polynom_order):
    weights_vec = []
    for i in range(0,polynom_order+1):
        weights_vec.append(rand.random())
    return weights_vec


def compute_new_weights(phi, predicted_values):
    tmp = np.linalg.inv(np.matmul(np.transpose(phi),phi))
    tmp = np.matmul(tmp,np.transpose(phi))
    return np.matmul( tmp,predicted_values )
        

data_set = generate_data_set()
polynom_order = 2
print(data_set)
matrixA = np.array(generate_general_A_matrix(polynom_order, data_set) )
print("A matrix:")
print(matrixA)
  
weightVec = generate_weights_vector(polynom_order)    
print("Initial weights:")
print(weightVec)

bVec = compute_b_matrix(matrixA, weightVec)
print("B Vector:")
print(bVec)

phi = np.array(computed_phi(data_set, polynom_order) )
print("Phi:")
print(phi)

Vec_T = predict_t_value(phi, weightVec)
print("T Vector")
print(Vec_T)

B_temp = np.matmul((np.transpose(phi)),Vec_T)
print("B tmppp")
print(B_temp)


newWeightVec = compute_new_weights(phi, Vec_T)
print("New Weight")
print(newWeightVec)

        