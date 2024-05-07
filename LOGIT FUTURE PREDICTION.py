import pandas as pd

dataset = pd.read_csv('/Users/bunny2020/Downloads/logit classification.csv')
dataset

X = dataset.iloc[:,[2,3]].values

Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y, test_size=0.20,random_state=0)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty='l1' , solver = 'liblinear')
classifier.fit(X_train , Y_train)

y_pred = classifier.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
cm


from sklearn.metrics import accuracy_score
ar = accuracy_score(Y_test, y_pred)
ar

from sklearn.metrics import classification_report
cr = classification_report(Y_test, y_pred)
cr


bias = classifier.score(X_train , Y_train)
bias


varience = classifier.score(X_test , Y_test)
varience







dataset1 = pd.read_csv('/Users/bunny2020/Downloads/Future prediction1.csv')
dataset1


d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred = pd.DataFrame()

d2['y_pred'] = classifier.predict(M)
d2.to_csv('final.csv')


import os
os.getcwd()



