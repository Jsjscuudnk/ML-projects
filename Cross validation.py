

import pandas as pd

data = pd.read_csv('//Users/bunny2020/Downloads/Social_Network_Ads.csv')

X = data.iloc[:,[2,3]].values
Y = data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X,)

from sklearn.model_selection import train_test_split
X_test , X_train , Y_test, Y_train = train_test_split(X ,Y ,test_size=0.2 , random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train , Y_train)
y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(Y_test, y_pred)
ac

bais = knn.score(X_train , Y_train)
bais

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = knn, X = X_train , y = Y_train , cv = 5)
acc
acc.mean()*100
