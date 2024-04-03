# Implement Linear Regression and calculate sum of residual error on the following 
# Datasets. 
#     x = [0, 1, 2, 3, 4, 5, 6, 7, 8,   9] 
#     y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12] 
#   Implement gradient descent (both Full-batch and Stochastic with stopping
# criteria) on Least Mean Square loss formulation to compute the coefficients of
# regression matrix and compare the results using performance measures such as R2
# SSE etc.

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]

import numpy as np

# Cost function
def cost_function(X, y, theta):
    m = len(X)
    predictions = X.dot(theta)
    sq_errors = (predictions - y) ** 2
    return 1/(2*m) * sq_errors.sum()

# Gradient descent function
def gradient_descent(X, y, theta, learning_rate, num_iterations, method='full-batch'):
    m = len(X)
    for i in range(num_iterations):
        if method == 'full-batch':
            predictions = X.dot(theta)
            gradients = (1/m) * X.T.dot(predictions - y)
        elif method == 'stochastic':
            for j in range(m):
                random_index = np.random.randint(0, m)
                x_j = X[random_index]
                y_j = y[random_index]
                prediction = x_j.dot(theta)
                gradient = (prediction - y_j) * x_j
                theta -= learning_rate * gradient
        else:
            raise ValueError("Invalid method. Choose either 'full-batch' or 'stochastic'.")
        if i % 100 == 0:
            print(f"Iteration {i}: cost = {cost_function(X, y, theta)}")
    return theta

# Initialize theta
theta = np.zeros(1)

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent with full-batch method
theta_full_batch = gradient_descent(np.array(x).reshape((10, 1)), np.array(y), theta, learning_rate, num_iterations, method='full-batch')

# Perform gradient descent with stochastic method
theta_stochastic = gradient_descent(np.array(x).reshape((10, 1)), np.array(y), theta, learning_rate, num_iterations, method='stochastic')

print(f"Theta (full-batch): {theta_full_batch}")
print(f"Theta (stochastic): {theta_stochastic}")

