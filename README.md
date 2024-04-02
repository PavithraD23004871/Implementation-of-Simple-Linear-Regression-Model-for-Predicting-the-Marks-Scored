![image](https://github.com/PavithraD23004871/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955967/c4242608-9774-4c54-b853-2aaf9707cbcc)![image](https://github.com/PavithraD23004871/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955967/38806c6a-5c2c-454c-8b0a-e529cf75a5ca)# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Pandas as pd & Import numpy as np
2. Calulating The y_pred & y_test
3. Find the graph for Training set & Test Set
4. Find the values of MSE,MSA,RMSE


## Program:
``````
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: pavithra D
RegisterNumber:  212223230146

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/Book.csv')
df.head(10)

plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')

x=df.iloc[:,0:1]
y=df.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

x_train
y_train

lr.predict(x_test.iloc[0].values.reshape(1,1))

plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')
``````
## Output:
1.HEAD:

![image](https://github.com/PavithraD23004871/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955967/e7dc94fd-6584-44d8-8afd-4b524f0b13f1)


# plot:

![image](https://github.com/PavithraD23004871/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955967/d8d4c039-41de-4074-94be-a988bdedd327)


# iloc:

![image](https://github.com/PavithraD23004871/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955967/c5a85d2e-c5ad-48be-b161-9f37138bb08d)


# Training:
![image](https://github.com/PavithraD23004871/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955967/2643e0b7-1e9b-40b0-9a95-ad9dc2ff5e6c)


# Graph:

![image](https://github.com/PavithraD23004871/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138955967/1c394d49-82c1-4739-8983-3882d7c6e1e8)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
