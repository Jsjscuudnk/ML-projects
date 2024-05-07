import pandas as pd

import numpy as np

data = pd.read_csv('/Users/bunny2020/Downloads/Churn_Modelling.csv')

X = data.iloc[:,3:-1].values
Y = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct  = ColumnTransformer(transformers=[('encoders', OneHotEncoder() , [1])] ,remainder='passthrough')
X = np.array(ct.fit_transform(X))


from sklearn.model_selection import train_test_split
X_test,X_train,Y_test,Y_train = train_test_split(X,Y, test_size=0.3 , random_state=0)



from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth = 20 , n_estimators = 200 , learning_rate = 0.01)
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
