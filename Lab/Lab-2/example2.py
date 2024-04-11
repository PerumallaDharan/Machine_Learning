import numpy as np
import pandas as pd

# Example 02: Univariate analysis

# Importing the necessary libraries
from sklearn import datasets  # Importing the database
import matplotlib.pyplot as plt

# Loading the iris dataset
iris = datasets.load_iris()

# Converting the dataset to a dataframe
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['species'])

# Replacing the target values with class labels
iris.species = np.where(iris.species == 0.0, 'setosa', np.where(iris.species == 1.0, 'versicolor', 'virginica'))

# Removing spaces from column names
iris.columns = iris.columns.str.replace(' ', '')

# Displaying the summary statistics of the dataset
iris.describe()

# The 'species' column is categorical, so let's check the frequency distribution for each category.

# Printing the frequency distribution of each category in the 'species' column
print(iris['species'].value_counts())