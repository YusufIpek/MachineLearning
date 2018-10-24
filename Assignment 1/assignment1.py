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

#TODO  make the predicate function
   
def predict_t_value(phi,weightsVec):
    return np.matmul(np.transpose(phi), weightsVec)

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
    result = []
    for i in range(0,order+1):
        sum_val = 0
        for x in x_values:
            sum_val += x**i
        result.append(sum_val)
    return result


def generate_weights_vector(polynom_order):
    weights_vec = []
    for i in range(0,polynom_order+1):
        weights_vec.append(rand.random())
    return weights_vec


def compute_new_weights(phi, predicted_values):
    tmp = np.linalg.inv(np.matmul(np.transpose(phi),phi))
    tmp = np.matmul(tmp,np.transpose(phi))
    return tmp * predicted_values
        

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

phi = computed_phi(data_set, polynom_order)
print("Phi:")
print(phi)

scalarT = predict_t_value(phi, weightVec)
print("T Scalar")
print(scalarT)

print("B tmppp")
print((np.transpose(phi))*scalarT)


newWeightVec = compute_new_weights(phi, scalarT)
print("New Weight")
print(newWeightVec)

        