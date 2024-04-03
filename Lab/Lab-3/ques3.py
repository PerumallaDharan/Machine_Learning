import numpy as np


def find_correlation(vector1, vector2):
    # COMPUTE CORRELATION COEFFICIENT
    correlation = np.corrcoef(vector1, vector2)[0][1]

    return correlation


# EXAMPLE FEATURE VECTORS
N = int(input("Enter the dimension N for the feature vectors: "))
vector1 = np.random.rand(N)  
vector2 = np.random.rand(N)  

# FINDING CORRELATION
correlation = find_correlation(vector1, vector2)

print("Correlation between Vector 1 and Vector 2:", correlation)