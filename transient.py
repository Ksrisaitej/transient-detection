import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
trained_data = pd.read_csv('transient_detection_train.csv')

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
loop = 1

while loop:
    print("\n" + "=" * 50)
    print("TRANSIENT EVENT PREDICTION")
    print("=" * 50)

    # Collect inputs
    brightness_change_rate = float(input("Enter brightness change rate: "))
    flux_variability = float(input("Enter flux variability: "))
    peak_brightness = float(input("Enter peak brightness: "))
    duration = float(input("Enter duration: "))
    spectral_index = float(input("Enter spectral index: "))
    rise_time = float(input("Enter rise time: "))
    decay_time = float(input("Enter decay time: "))
    lightcurve_skewness = float(input("Enter light curve skewness: "))

    # Create feature array (IMPORTANT: reshape for single prediction)
    features = np.array([[brightness_change_rate, flux_variability, peak_brightness,
                          duration, spectral_index, rise_time, decay_time,
                          lightcurve_skewness]])

    # Scale the features (CRITICAL for SVM!)
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = clf.predict(features_scaled)[0]

    # Display result
    print("\n" + "-" * 50)
    if prediction == 1:
        print("✓ PREDICTION: TRANSIENT EVENT")
    else:
        print("✓ PREDICTION: NON-TRANSIENT EVENT")
    print("-" * 50)

    # Ask for another prediction
    op = input("\nDo you wish to make another prediction? (Y/N): ").strip().upper()

    if op == "Y":
        loop = 1
    elif op == "N":
        loop = 0
        print("\nThank you for using the prediction system!")
    else:
        print("Invalid input. Exiting...")
        loop = 0
