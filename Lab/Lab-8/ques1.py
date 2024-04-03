# Consider  the  two  dimensional  data  matrix  [(2,  1),  (3,  4),  (5,  0),  (7,  6),  (9,  2)]. 
# Implement principal component analysis. Use this to obtain the feature in transformed 
# 2D  feature  space.  Plot  the  scatter  plot  of  data  points  in  both  the  original  as  well  as 
# transformed domain.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Given data matrix
data = np.array([[2, 1], [3, 4], [5, 0], [7, 6], [9, 2]])

# Instantiate PCA
pca = PCA(n_components=2)

# Fit PCA to the data
pca.fit(data)

# Transform the data to its principal components
transformed_data = pca.transform(data)

# Plot original data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.title('Original Data')
plt.xlabel('X')
plt.ylabel('Y')

# Plot transformed data
plt.subplot(1, 2, 2)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color='red')
plt.title('Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
