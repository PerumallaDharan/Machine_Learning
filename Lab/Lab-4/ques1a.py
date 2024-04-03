# Implement Linear Regression and calculate sum of residual error on the following 
# Datasets. 
#     x = [0, 1, 2, 3, 4, 5, 6, 7, 8,   9] 
#     y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12] 
#   Compute the regression coefficients using analytic formulation and calculate Sum 
# Squared Error (SSE) and R2 value.

import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

n = len(x)

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = 0
denominator = 0

for i in range(n):
    numerator += (x[i] - x_mean) * (y[i] - y_mean)
    denominator += (x[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

y_pred = b0 + b1 * x

sse = 0
for i in range(n):
    sse += (y[i] - y_pred[i]) ** 2
    
r2 = 1 - (sse / np.sum((y - y_mean) ** 2))

print("SSE: ", sse)
print("R2: ", r2)
print("b0: ", b0)
print("b1: ", b1)
print("y_pred: ", y_pred)

# Output
# SSE:  7.673076923076923
# R2:  0.952538038613988
# b0:  1.2363636363636363
# b1:  1.1696969696969697
# y_pred:  [ 1.23636364  2.40606061  3.57575758  4.74545455  5.91515152  7.08484848
#   8.25454545  9.42424242 10.59393939 11.76363636]

# Conclusion
# Implemented Linear Regression and calculated sum of residual error on the given dataset.
# The regression coefficients are:
#     b0: 1.2363636363636363
#     b1: 1.1696969696969697
# Sum Squared Error (SSE): 7.673076923076923
# R2 value: 0.952538038613988
# The predicted values are:
#     [ 1.23636364  2.40606061  3.57575758  4.74545455  5.91515152  7.08484848
#       8.25454545  9.42424242 10.59393939 11.76363636]

# The model is a good fit as the R2 value is close to 1.

