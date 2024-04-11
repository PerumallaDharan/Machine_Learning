from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the file path to the dataset on your laptop
dataset_path = "sports.csv"  # Replace this with the actual file path

# Load the dataset from CSV
df = pd.read_csv(dataset_path)
 
# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Outlook', 'Temp.', 'Humidity', 'Wind'])

# Split features and target
X = df.drop('Decision', axis=1)  # 'Decision' is the target column
y = df['Decision']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naïve Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Compute other classification measures
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
