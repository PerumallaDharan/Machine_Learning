# Implement Principal Component Analysis Algorithm and use it to reduce dimensions 
# of  Iris  Dataset  (from  4D  to  2D).  Plot  the  scatter  plot  for  samples  in  the  transformed 
# domain with different colour codes for samples belonging to different classes.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load Iris dataset from file
data = np.genfromtxt('E:\SRM\Machine_Learning\Lab\Lab-8\iris.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2, 3))
targets = np.genfromtxt('E:\SRM\Machine_Learning\Lab\Lab-8\iris.csv', delimiter=',', skip_header=1, usecols=4, dtype=str)

# Implement PCA algorithm
def pca(X, n_components=2):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components eigenvectors
    W = eigenvectors[:, :n_components]
    
    # Project the data onto the new feature space
    transformed_data = np.dot(X_centered, W)
    
    return transformed_data

# Reduce dimensions of Iris dataset using PCA
X_transformed = pca(data, n_components=2)

# Plot scatter plot for samples in the transformed domain with different color codes
plt.figure(figsize=(8, 6))
targets_unique = np.unique(targets)
colors = ['r', 'g', 'b']
for i, target in enumerate(targets_unique):
    plt.scatter(X_transformed[targets == target, 0], X_transformed[targets == target, 1], c=colors[i], label=target)

plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
