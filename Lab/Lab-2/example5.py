# Example 05: Multivariate analysis

import seaborn as sns

import matplotlib.pyplot as plt

# Load the iris dataset
iris = sns.load_dataset('iris')
# Print the mean of each feature for each label class
print(iris.groupby(by="species").mean())

# Plot the mean of each feature for each label class as a bar chart
iris.groupby(by="species").mean().plot(kind="bar")

# Set the title of the plot
plt.title('Class vs Measurements')

# Set the label for the y-axis
plt.ylabel('mean measurement(cm)')

# Rotate the x-axis tick labels to improve readability
plt.xticks(rotation=0)

# Enable grid lines on the plot
plt.grid(True)

# Place the legend outside the plot area for tidiness
plt.legend(loc="upper left", bbox_to_anchor=(1,1))