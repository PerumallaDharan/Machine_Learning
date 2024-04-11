# Example 06: Correlation matrix

import seaborn as sns

import matplotlib.pyplot as plt

# Load the iris dataset
iris = sns.load_dataset('iris')

# # create correlation matrix
# corr = iris.corr()
# print(corr)
import statsmodels.api as sm
# sm.graphics.plot_corr(corr, xnames=list(corr.columns))
# plt.show()

# Exclude the 'species' column
numerical_iris = iris.drop('species', axis=1)

# Create correlation matrix
corr = numerical_iris.corr()
print(corr)

# Plot the correlation matrix
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()