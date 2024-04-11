#AP21110010240
#G.Dinesh
 
# 1. Implement Decision Tree Classifier for classification of Iris dataset
# a. Load the data set
# b. Split the data set to train and test sets
# c. Train a Decision Tree using train set
# d. Test the model using test set. Find accuracy and confusion Matrix.



# Set the file path to the Iris dataset on the desktop
iris_data_path = "E:\SRM\Machine_Learning\Lab\Lab-6\iris.csv"  # Replace this with the actual file path

# Load the Iris dataset from CSV
import pandas as pd
iris_df = pd.read_csv(iris_data_path)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset from CSV
iris_df = pd.read_csv(iris_data_path)

# Split features and target
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, columns=iris_df['Species'].unique(), index=iris_df['Species'].unique())
print("\nConfusion Matrix:")
print(conf_matrix_df)

# Print the first few rows and column names of the DataFrame
print("DataFrame Head:")
print(iris_df.head())
print("\nColumns:", iris_df.columns)
