# Import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
def dataset3Params(X, y, Xval, yval,vals):
    """
    dataset3Params returns the optimal C and gamma(1/sigma) based on a cross-validation set.
    """
    acc = 0
    best_C = 0
    best_gamma = 0
    for i in vals:
        C= i
        for j in vals:
            gamma = 1/j
            classifier = SVC(C=C, gamma=gamma)
            classifier.fit(X, y)
            prediction = classifier.predict(Xval)
            score = classifier.score(Xval, yval)
            print("C= ",C,", sigma= ",j,", score= ",score)
            if score > acc:
                acc = score
                best_C = C
                best_gamma = gamma
    return best_C, best_gamma


data_train = pd.read_csv("data_train.data", header=None)
data_val = pd.read_csv("data_val.data", header=None)
data_test = pd.read_csv("data_test.data", header=None)

X_train = data_train.values[:, :-1]
y_train = data_train.values[:, 25]
X_val = data_val.values[:, :-1]
y_val = data_val.values[:, 25]
X_test = data_test.values[:, :-1]
y_test = data_test.values[:, 25]

vals = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50, 70]
best_C, best_gamma = dataset3Params(X_train, y_train.ravel(), X_val, y_val.ravel(), vals)

#What are the best C and sigma ?
print("Best C: ",best_C)
print("Best sigma: ",1/best_gamma)

#Build an SVM classifier with the best C and gamma and get classifier score of about 95% ?
classifier = SVC(C=best_C,gamma=best_gamma, kernel="linear")#, kernel="poly", degree=3, coef0=10.0)
classifier.fit(X_train,np.ravel(y_train))
print(float(classifier.score(X_train,y_train)))


def learning_curve():
    pass
