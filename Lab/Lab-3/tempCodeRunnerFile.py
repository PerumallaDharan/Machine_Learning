import numpy as np


def find_mean_and_variance(feature_vector):
    # FINDING MEAN
    mean = np.mean(feature_vector)

    # FINDING COVARIANCE
    variance = np.var(feature_vector)

    return mean, variance


# TAKING INPUT FROM THE USER FOR THE FEATURE VECTOR
n = int(input("Enter the number of elements in the feature vector: "))
feature_vector = np.zeros(n)
for i in range(n):
    feature_vector[i] = float(input(f"Enter element {i + 1}: "))

# FINDING MEAN AND VARIANCE
mean, variance = find_mean_and_variance(feature_vector)

print("Mean:", mean)
print("Variance:", variance)
