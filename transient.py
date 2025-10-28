import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
trained_data = pd.read_csv('PS2_transient_detection_train.csv')

X = trained_data.drop(['label'], axis=1)
Y = trained_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
from time import time
t0 = time()
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print("Training time for linear kernel: ", round(time() - t0,3))
t0 = time()
y_pred1 = clf.predict(X_test)
print("Prediction time: ", round(time() - t0,3))
t0 = time()
clf = SVC(kernel='rbf',C=1,gamma='scale')
clf.fit(X_train, y_train)
print("Training time for rbf kernel: ", round(time() - t0,3))
t0 = time()
y_pred2 = clf.predict(X_test)
print("Prediction time: ", round(time() - t0,3))

from sklearn.metrics import accuracy_score

accuracy1 = accuracy_score(y_test, y_pred1)
print("Accuracy for linear kernel: ", round(accuracy1,3))
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy for rbf kernel: ", round(accuracy2,3))

print("\n" + "=" * 60)



