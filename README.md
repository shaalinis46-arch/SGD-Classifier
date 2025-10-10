# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the Iris dataset.
2. Split the dataset into training and testing sets.
3. Standardize the input features for better model performance.
4. Train the SGD classifier and predict the species of the Iris flower.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: shaalini.s
RegisterNumber: 25017649 
*/
# Logistic Regression using SGDClassifier

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data          # Features (sepal length, sepal width, petal length, petal width)
y = iris.target        # Target (species)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SGD Classifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict the species for the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Predicted Species:", y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
```

## Output:
<img width="749" height="250" alt="Screenshot 2025-10-10 152940" src="https://github.com/user-attachments/assets/51432a1d-28c1-421d-b9f6-2c5da7d4370c" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
