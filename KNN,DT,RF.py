import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

data = pd.read_csv('/Users/bunny2020/Downloads/emp_sal.csv')
data

X = data.iloc[:,1:2].values
Y = data.iloc[:,2].values

from sklearn.svm import SVR

regressor = SVR(kernel='poly' , degree=5 , gamma='scale')
regressor.fit(X , Y)
y_pred_svr = regressor.predict([[7.5]])
y_pred_svr


from sklearn.neighbors import KNeighborsRegressor

regressor_knn = KNeighborsRegressor(n_neighbors=2 , weights='distance' , algorithm='ball_tree')
regressor_knn.fit(X,Y)
y_pred_knn = regressor_knn.predict([[7.5]])
y_pred_knn


from sklearn.tree import DecisionTreeRegressor

regressor_dt = DecisionTreeRegressor(criterion='poisson' , splitter='best' , random_state=0 , max_depth= 4)
regressor_dt.fit(X , Y)
y_pred__dt = regressor_dt.predict([[7.5]])
y_pred__dt


from sklearn.ensemble import RandomForestRegressor

regressor_rf = RandomForestRegressor(n_estimators=30 , random_state=0)
regressor_rf.fit(X , Y)
y_pred_rf = regressor_rf.predict([[7.5]])
y_pred_