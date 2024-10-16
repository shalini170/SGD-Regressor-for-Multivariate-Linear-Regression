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
'''
![Screenshot 2024-10-16 070611](https://github.com/user-attachments/assets/581c24f4-98a7-4e1f-9a18-15e7a2acffe1)



```

## Output:
'''
![Screenshot 2024-10-16 070340](https://github.com/user-attachments/assets/771a04d3-5a2d-40b3-ae46-90d32b36eccb)
'''



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
