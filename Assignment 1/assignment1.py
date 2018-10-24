import numpy as np
import random as rand

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
            if i == 0 and j == 0:
                tmp.append(len(x_values))
            else:
                tmp.append(sigma(i+j, x_values))
        A.append(tmp)
    return A

#TODO  make the predicate function
    
def generate_general_b_matrix(predicted_func, x_values):
    b = []
    for it in range(0,len(x_values)):
        res = 0
        for value in x_values:
            res += predicted_func(value) * value**it
        b.append(res)

data_set = generate_data_set()

print(data_set)
res = generate_general_A_matrix(1, data_set)
for row in res:
    print(row)
        