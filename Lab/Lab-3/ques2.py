import numpy as np


def find_covariance(vector1, vector2):
    # FINDING COVARIANCE
    covariance = np.cov(vector1, vector2)[0][1]

    return covariance


M = int(input("Enter the dimension M for the feature vectors: "))
vector1 = np.random.rand(M)  
vector2 = np.random.rand(M)  

# FINDING COVARIANCE
covariance = find_covariance(vector1, vector2)

print("Covariance:", covariance)
