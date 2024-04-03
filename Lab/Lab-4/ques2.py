# Download Boston Housing Rate Dataset. Analyse the input attributes and find out the
# attribute that best follow the linear relationship with the output price. Implement both the
# analytic formulation and gradient descent (Full-batch, stochastic) on LMS loss
# formulation to compute the coefficients of regression matrix and compare the results.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Full path to the housing dataset file
file_path = r'E:\SRM\Machine Learning\Lab\Lab-4\BostonHousing.csv'

# Load the housing dataset from the file
housing_data = pd.read_csv(file_path)

# Display the first few rows and column names of the dataset
print(housing_data.head())
print("Column names:", housing_data.columns)

# Select input attributes (features) and output (price) for analysis
X = housing_data.drop('medv', axis=1)  # Corrected to 'medv'
y = housing_data['medv']

# Calculate correlation coefficients between input attributes and output price
correlations = X.corrwith(y)
best_attribute = correlations.abs().idxmax()

# Plot the best attribute against the output price to visualize the linear relationship
plt.scatter(X[best_attribute], y)
plt.xlabel(best_attribute)
plt.ylabel('Price')
plt.title('Relationship between Input Attribute and Price')
plt.show()

# Add bias term to input features
X_b = np.c_[np.ones((len(X), 1)), X]

# Analytic formulation for linear regression
coefficients_analytic = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Coefficients using analytic formulation:")
print(coefficients_analytic)

# Implement gradient descent (Full-batch)
def gradient_descent_full_batch(X, y, learning_rate=0.01, num_iterations=1000):
    m = len(X)
    n = X.shape[1]
    theta = np.random.randn(n, 1)  # Initialize coefficients randomly
    for iteration in range(num_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients
    return theta

coefficients_gradient_full_batch = gradient_descent_full_batch(X_b, y)
print("Coefficients using gradient descent (full-batch):")
print(coefficients_gradient_full_batch)

# Implement stochastic gradient descent
def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=50):
    m = len(X)
    n = X.shape[1]
    theta = np.random.randn(n, 1)  # Initialize coefficients randomly
    for epoch in range(num_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradients
    return theta

coefficients_stochastic_gradient = stochastic_gradient_descent(X_b, y)
print("Coefficients using stochastic gradient descent:")
print(coefficients_stochastic_gradient)
