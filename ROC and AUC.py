import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('//Users/bunny2020/Downloads/Social_Network_Ads.csv')

X =  data.iloc[:,[2,3]].values
Y = data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split

X_test , X_train , Y_test , Y_train = train_test_split(X ,Y, test_size=0.2 , random_state=0)

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

bias = classifier.score(X_train , Y_train)
bias

from sklearn.metrics import roc_curve, auc
fpr, tpr, specificity = roc_curve(Y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
