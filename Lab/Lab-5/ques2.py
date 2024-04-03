# Import the Iris dataset. Write a program to obtain the Euclidian Distance Matrix for
# all the data samples in the feature space. Distance metric is a 2D array, where the
# (i,j)th entry represents the distance between the ith and jth sample points in the feature
# space.

import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Load the Iris dataset
file_path = r'E:\SRM\Machine Learning\Lab\Lab-5\iris.csv'
iris_data = pd.read_csv(file_path)

# Extract features (attributes) from the dataset
X = iris_data.iloc[:, :-1]  # Exclude the last column (species)

# Calculate pairwise Euclidean distances
distances = pdist(X, metric='euclidean')

# Convert pairwise distances to a square distance matrix
euclidean_distance_matrix = squareform(distances)

# Print the Euclidean Distance Matrix
print("Euclidean Distance Matrix:")
print(euclidean_distance_matrix)
