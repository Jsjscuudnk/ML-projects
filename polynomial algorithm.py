import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/bunny2020/Downloads/emp_sal.csv')
dataset

X = dataset.iloc[:,1:2].values

Y = dataset.iloc[:,2].values



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=5)

X_poly = poly_reg.fit_transform(X)


poly_reg.fit(X_poly , Y)

lin_reg2 = LinearRegression()

lin_reg2.fit(X_poly , Y)





plt.scatter(X , Y , color = 'red')
plt.plot(X , lin_reg.predict(X) , color = 'blue')
plt.xlabel('Position Value')
plt.ylabel('Salary')
plt.show()



plt.scatter(X , Y , color = 'red')
plt.plot(X , lin_reg2.predict(poly_reg.fit_transform(X)) , color = 'blue')
plt.xlabel('Position Value')
plt.ylabel('Salary')
plt.show()

lin_pred = lin_reg.predict([[7.5]])
lin_pred

poly_pred = lin_reg2.predict(poly_reg.fit_transform([[7.5]]))
poly_pred








