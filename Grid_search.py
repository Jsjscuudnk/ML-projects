import pandas as pd

data = pd.read_csv('//Users/bunny2020/Downloads/Social_Network_Ads.csv')

X = data.iloc[:,[2,3]].values
Y = data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X,)

from sklearn.model_selection import train_test_split
X_test , X_train , Y_test, Y_train = train_test_split(X ,Y ,test_size=0.2 , random_state=0)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train , Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(Y_test, y_pred)
ac

bais = classifier.score(X_train , Y_train)
bais
from sklearn.model_selection import GridSearchCV
parameters = {'C':[1,2,34,50,100] , 'kernel':['poly' , 'rbf' , 'linear'] , 'gamma':[0.1,0.2,0.3,0.4]}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters , cv = 5)
grid_search.fit(X_train ,Y_train)
best_params = grid_search.best_params_
best_params
best_score = grid_search.best_score_
best_score
