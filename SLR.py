import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/bunny2020/Downloads/Salary_Data.csv')
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
Y=Y.reshape(-1,1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test= train_test_split(X , Y , test_size=0.2 , random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train , Y_train)


y_pred = regressor.predict(X_test)


m = regressor.coef_
m


c = regressor.intercept_
c


y_12 = m*12+c
y_12


plt.scatter(X_test, Y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


