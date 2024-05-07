import pandas as pd

data = pd.read_csv('//Users/bunny2020/Downloads/Social_Network_Ads.csv')

X = data.iloc[:,[2,3]].values
Y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_test , X_train , Y_test , Y_train = train_test_split(X ,Y, test_size=0.3 , random_state=0)
'''
from sklearn.preprocessing import StandardScaler
sc  = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy' , max_depth=5 , random_state=0 , splitter='random')
dt.fit(X_train , Y_train)

y_pred = dt.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
cm

from sklearn.metrics import accuracy_score

ac = accuracy_score(Y_test, y_pred)
ac

bias = dt.score(X_train , Y_train)
bias

varience = dt.score(X_test , Y_test)
varience

