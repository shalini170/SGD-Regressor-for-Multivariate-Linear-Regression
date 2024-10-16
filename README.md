# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Data preparation
3. Hypothesis Definition
4. Cost Function
5.Parameter Update Rule
6.Iterative Training
7.Model evaluation
8.End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: shalini venkatesulu
RegisterNumber:  212223220104
*/
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Shruthi.S
RegisterNumber:  212222220044
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
![Screenshot 2024-10-16 065842](https://github.com/user-attachments/assets/c1cc857d-aaa9-4076-9bab-24eb1013edea)

X = df.drop(columns=['AveOccup','HousingPrice'])
X.info()
![Screenshot 2024-10-16 065951](https://github.com/user-attachments/assets/4c96436f-4ed9-49b5-b5c1-5902842c70bd)

Y = df[['AveOccup','HousingPrice']]
Y.info()
![Screenshot 2024-10-16 070050](https://github.com/user-attachments/assets/d3c9757c-3052-4916-962e-82d3ddee5cba)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train
![Screenshot 2024-10-16 070140](https://github.com/user-attachments/assets/b1f7c813-6072-47f2-8659-e9f80ec8b2e4)

Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

![Screenshot 2024-10-16 070238](https://github.com/user-attachments/assets/0ab52bc8-286a-48a4-bc13-5fba83b3e849)

print("\nPredictions:\n", Y_pred[:5])


```

## Output:
'''
![Screenshot 2024-10-16 070340](https://github.com/user-attachments/assets/771a04d3-5a2d-40b3-ae46-90d32b36eccb)
'''



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
