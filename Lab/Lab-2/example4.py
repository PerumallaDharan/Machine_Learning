import seaborn as sns

import matplotlib.pyplot as plt

# Load the iris dataset
iris = sns.load_dataset('iris')

# Plot histogram
sns.histplot(data=iris)
plt.suptitle("Histogram", fontsize=16)
plt.show()

# Plot boxplot
sns.boxplot(data=iris)
plt.title("Box Plot", fontsize=16)
plt.show()