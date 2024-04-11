# Example 01: Normalization and scaling

from sklearn import datasets  # Importing the datasets module from sklearn
import numpy as np  # Importing the numpy module as np
from sklearn import preprocessing  # Importing the preprocessing module from sklearn

iris = datasets.load_iris()  # Loading the iris dataset
X = iris.data[:, [2, 3]]  # Assigning the features to variable X
y = iris.target  # Assigning the target variable to variable y

std_scale = preprocessing.StandardScaler().fit(X)  # Creating a StandardScaler object and fitting it to the data
X_std = std_scale.transform(X)  # Transforming the data using the fitted scaler

minmax_scale = preprocessing.MinMaxScaler().fit(X)  # Creating a MinMaxScaler object and fitting it to the data
X_minmax = minmax_scale.transform(X)  # Transforming the data using the fitted scaler

