import numpy as np

def euclidean_distance(point1, point2):
    print(f"Euclidean distance for {point1} and {point2}:")
    distance = np.sqrt(np.sum((point1 - point2) ** 2))
    print(distance)
    return distance

def kmeans_clustering(data, k, initial_points=None):
    num_samples, num_features = data.shape
    
    if initial_points is None:
        # Randomly initialize centroids
        np.random.seed(0)
        centroids = data[np.random.choice(num_samples, k, replace=False)]
    else:
        centroids = np.array(initial_points)
    
    print("Initial Centroids:")
    print(centroids)
    
    old_centroids = np.zeros(centroids.shape)
    clusters = np.zeros(num_samples)
    distances = np.zeros((num_samples, k))
    error = euclidean_distance(centroids, old_centroids)
    
    iteration = 0
    while error != 0:
        # Assign each data point to the closest centroid
        for i in range(k):
            distances[:, i] = np.apply_along_axis(euclidean_distance, 1, data, centroids[i])
        
        clusters = np.argmin(distances, axis=1)
        
        # Save old centroids for convergence check
        old_centroids = np.copy(centroids)
        
        # Update centroids
        for i in range(k):
            centroids[i] = np.mean(data[clusters == i], axis=0)
        
        error = euclidean_distance(centroids, old_centroids)
        iteration += 1
        
        print(f"Iteration {iteration} - Centroids:")
        print(centroids)
        
        # Display final clustering result
        print("\nFinal Clusters:")
        for i in range(k):
            print(f"Cluster {i+1}:")
            for j, idx in enumerate(clusters):
                if idx == i:
                    print(f"ID: {j+1}, Cluster: {i+1}")
    


num_features = int(input("Enter the number of features: "))
num_samples = int(input("Enter the number of samples: "))
k = int(input("Enter the number of clusters: "))
    
data = np.zeros((num_samples, num_features))
for feature_index in range(num_features):
    print(f"Enter data for feature {feature_index + 1}:")
    for i in range(num_samples):
        data[i, feature_index] = float(input())
    
initial_option = input("Do you want to provide initial points? (y/n): ")
if initial_option.lower() == 'y':
    initial_points = []
    print("Enter initial points:")
    for _ in range(k):
        initial_point = []
        for feature_index in range(num_features):
            initial_point.append(float(input(f"Initial point for feature {feature_index + 1}: ")))
        initial_points.append(initial_point)
else:
    initial_points = None
    
kmeans_clustering(data, k, initial_points)