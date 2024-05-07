import pandas as pd

data = pd.read_csv('/Users/bunny2020/Downloads/Social_Network_Ads.csv')

X = data.iloc[:,[2,3]].values
Y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_test,X_train,Y_test,Y_train = train_test_split(X , Y , test_size=0.30, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


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

from sklearn.metrics import classification_report
cr = classification_report(Y_test , y_pred)
cr

bias = classifier.score(X_train , Y_train)
varince = classifier.score(X_test , Y_test)

