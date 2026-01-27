<img width="1378" height="227" alt="image" src="https://github.com/user-attachments/assets/e40120b5-de83-45c2-851f-2e626c73b837" /><img width="1378" height="227" alt="image" src="https://github.com/user-attachments/assets/979d47f6-6410-4f8a-a821-c2d367d408fb" /># Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2. Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3. Create a linear regression model and fit it using the training data.
4. Predict the results for the testing set and plot the training and testing sets with fitted lines. 
5. Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ROHITH R
RegisterNumber:  25002211
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df

X = df[["Hours_Studied"]] 
y = df["Marks_Scored"]  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()

hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")



```

## Output:
<img width="494" height="603" alt="Screenshot 2026-01-27 114434" src="https://github.com/user-attachments/assets/990c4534-fb8b-4aa9-85de-4acb59e09d18" />
<img width="482" height="194" alt="Screenshot 2026-01-27 114445" src="https://github.com/user-attachments/assets/d18559e4-dbce-4c21-b513-5999bebbd0c0" />
<img width="496" height="183" alt="Screenshot 2026-01-27 114453" src="https://github.com/user-attachments/assets/d3bac39f-7b61-43af-87f7-f76b872bf250" />
<img width="1206" height="687" alt="Screenshot 2026-01-27 114509" src="https://github.com/user-attachments/assets/d5365b63-54e7-4eaf-828b-903b2c1f3385" />
<img width="1378" height="227" alt="Screenshot 2026-01-27 114526" src="https://github.com/user-attachments/assets/b354e38a-6089-4323-8cd9-ba805a777f21" />








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
