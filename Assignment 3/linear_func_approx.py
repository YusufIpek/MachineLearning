import numpy as np 

def predict(x, weights):
    res = []
    i = 0
    val1 = 0
    val2 = 0
    for w in weights:  
        val1 += w[0] * x[0]**i
        val2 += w[1] * x[1]**i
        i += 1
    res.append(val1)
    res.append(val2)
    return res


def target_value(x, weights, gamma, R):
    predicted_value = predict(x,weights)
    return [(predicted_value[0] * gamma) + R, (predicted_value[1] * gamma) + R]

def cost_function(x, target, weights):
    N = len(target)
    prediction = predict(x, weights)
    sq_error = [(prediction[0] - target[0])**2,(prediction[1] - target[1])**2]
    return [1.0/(2*N) * sq_error[0], 1.0/(2*N) * sq_error[1]]

def update_weights_vectorized(x, target, weights, learning_rate):
    # See: https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
    prediction = predict(x, weights)
    error = [target[0] - prediction[0], target[1] - prediction[1]]
    gradient = [-x[0]*error[0], -x[1]*error[1]]

    gradient = [gradient[0] * learning_rate, gradient[1] * learning_rate]
    updated_weights = list(map(lambda w: [w[0]-gradient[0],w[1]-gradient[1]], weights))
    return updated_weights


gamma = 0.9
R = -1
learning_rate = 0.01
x = [0.2,0.8]
weights = []
for i in range(20):
    weights.append([1,1])


print("Input Feature:")
print(x)
print("Predicted Value:")
print(predict(x, weights))
print("Target Value:")
print(target_value(x, weights, gamma, R))
print("Squared Error Loss:")
print(cost_function(x, target_value(x,weights, gamma, R), weights))
print("Update Weights:")
updated_weights = update_weights_vectorized(x, target_value(x,weights,gamma,R),weights, learning_rate)
print(updated_weights)



x2 = [0.2,0.9]
R= 0.5

print("Predicted Value:")
print(predict(x2, updated_weights))
print("Target Value:")
print(target_value(x2, updated_weights, gamma, R))
print("Squared Error Loss:")
print(cost_function(x2, target_value(x2,updated_weights, gamma, R), updated_weights))
print("Update Weights:")
updated_weights = update_weights_vectorized(x2, target_value(x2,updated_weights,gamma,R),updated_weights, learning_rate)
print(updated_weights)


x3 = [0.4,0.8]
R=-1

print("Predicted Value:")
print(predict(x3, updated_weights))
print("Target Value:")
print(target_value(x3, updated_weights, gamma, R))
print("Squared Error Loss:")
print(cost_function(x3, target_value(x2,updated_weights, gamma, R), updated_weights))
print("Update Weights:")
updated_weights = update_weights_vectorized(x3, target_value(x3,updated_weights,gamma,R),updated_weights, learning_rate)
print(updated_weights)
