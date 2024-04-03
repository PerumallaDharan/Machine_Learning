import numpy as np


def compute_covariance_matrix(feature_matrix):
    # Compute covariance matrix
    covariance_matrix = np.cov(feature_matrix)

    return covariance_matrix


def compute_correlation_matrix(feature_matrix):
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(feature_matrix)

    return correlation_matrix


# Example feature matrix dimensions
M = int(input("Enter the dimension M for the feature vectors: "))
N = int(input("Enter the number of samples N: "))

# Generate a random feature matrix of dimension MxN
feature_matrix = np.random.rand(M, N)

# Compute covariance matrix
covariance_matrix = compute_covariance_matrix(feature_matrix)

# Compute correlation matrix
correlation_matrix = compute_correlation_matrix(feature_matrix)

print("Covariance Matrix:")
print(covariance_matrix)

print("\nCorrelation Matrix:")
print(correlation_matrix)
