# Import the Iris dataset. Prepare a dataset considering samples belong to any two
# output classes. Draw the scatter plot for all the samples in the new dataset considering
# any two input attributes. Examine the scatter plot to find the equation of a line that
# can separate sample of two classes.

import pandas as pd
import matplotlib.pyplot as plt

# Read the Iris dataset from CSV
iris_df = pd.read_csv('E:\SRM\Machine Learning\Lab\Lab-5\iris.csv')

# Select two classes from the dataset
class1 = 'setosa'
class2 = 'versicolor'

# Filter the dataset to include only the selected classes
selected_df = iris_df[(iris_df['Species'] == class1) | (iris_df['Species'] == class2)]

# Select two input attributes for scatter plot
attribute1 = 'Sepal.Length'
attribute2 = 'Sepal.Width'

# Plot the scatter plot for the selected classes and attributes
plt.scatter(selected_df[selected_df['Species'] == class1][attribute1],
            selected_df[selected_df['Species'] == class1][attribute2],
            label=class1)

plt.scatter(selected_df[selected_df['Species'] == class2][attribute1],
            selected_df[selected_df['Species'] == class2][attribute2],
            label=class2)

plt.xlabel(attribute1)
plt.ylabel(attribute2)
plt.title(f'Scatter Plot of {attribute1} vs {attribute2} for {class1} and {class2}')
plt.legend()
plt.show()

