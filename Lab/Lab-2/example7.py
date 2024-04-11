
import seaborn as sns

import matplotlib.pyplot as plt
# Load the iris dataset
iris = sns.load_dataset('iris')

# from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix
scatter_matrix(iris, figsize=(10, 10))
# use suptitle to add title to all sublots
plt.suptitle("Pair Plot", fontsize=20)