import pandas as pd

data = pd.read_csv('/Users/bunny2020/Downloads/Social_Network_Ads.csv')

X = data.iloc[:,[2,3]].values
Y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_test , X_train , Y_test , Y_train = train_test_split(X , Y, test_size=0.3 , random_state=0)

from sklearn.ensemble import RandomForestClassifier
r_ft = RandomForestClassifier(criterion='entropy' ,random_state=0,max_depth=5)
r_ft.fit(X_train , Y_train)
y_pred = r_ft.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm


from sklearn.metrics import accuracy_score
ac = accuracy_score(Y_test, y_pred)
ac


bais = r_ft.score(X_train, Y_train)
bais

varience = r_ft.score(X_test , Y_test)
varience
