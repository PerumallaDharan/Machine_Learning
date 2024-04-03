# Take two 3D coordinates from the user. Find out the distance between these points

# x and y are vectors representing the coordinates of two points.
# Here, k=3

import pandas as pd
from scipy.spatial import distance

# Load the Iris dataset
file_path = r'E:\SRM\Machine Learning\Lab\Lab-5\iris.csv'
iris_data = pd.read_csv(file_path)

# Select two random rows (data points) from the dataset
point1 = iris_data.sample(1, random_state=42).iloc[0, :-1].values  # Exclude the last column (species)
point2 = iris_data.sample(1, random_state=99).iloc[0, :-1].values  # Exclude the last column (species)

# Calculate Manhattan distance
manhattan_dist = distance.cityblock(point1, point2)

# Calculate Euclidean distance
euclidean_dist = distance.euclidean(point1, point2)

# Calculate Minkowski distance (with p=3)
minkowski_dist = distance.minkowski(point1, point2, p=3)

# Print the calculated distances
print(f"Manhattan distance: {manhattan_dist}")
print(f"Euclidean distance: {euclidean_dist}")
print(f"Minkowski distance (p=3): {minkowski_dist}")
