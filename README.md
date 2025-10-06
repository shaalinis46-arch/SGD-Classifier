# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: 
RegisterNumber:  
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Example: Predict Placement (1 = Placed, 0 = Not Placed)
data = {
    'CGPA': [8.5, 6.8, 7.9, 5.4, 9.1, 8.0, 7.5, 6.0, 9.3, 5.8],
    'Aptitude_Score': [82, 55, 75, 48, 92, 77, 73, 50, 95, 45],
    'Communication_Skill': [8, 6, 7, 5, 9, 8, 7, 6, 9, 5],
    'Placed': [1, 0, 1, 0, 1, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")


X = df[['CGPA', 'Aptitude_Score', 'Communication_Skill']]
y = df['Placed']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = SGDClassifier(loss='log_loss', max_iter=1000, learning_rate='optimal', random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

new_student = np.array([[8.3, 80, 8]])  # [CGPA, Aptitude, Communication]
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)

print("\nNew Student Prediction:")
if prediction[0] == 1:
    print(" The student is LIKELY to be PLACED.")
else:
    print("The student is NOT likely to be placed.")
*/
```

## Output:
![prediction of iris species using SGD Classifier](sam.png)
![alt text](<Screenshot 2025-10-06 211733.png>)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
