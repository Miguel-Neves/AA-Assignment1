import pandas as pd
import numpy as np

from LogReg_algorithm import logreg_algorithm
from SVM_algorithm import svm_algorithm

data_train = pd.read_csv("data_train.data", header=None)
data_val = pd.read_csv("data_val.data", header=None)
data_test = pd.read_csv("data_test.data", header=None)

X_train = data_train.values[:, :-1]
y_train = data_train.values[:, 25]
X_val = data_val.values[:, :-1]
y_val = data_val.values[:, 25]
X_test = data_test.values[:, :-1]
y_test = data_test.values[:, 25]

# SVM algorithm
vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
kernel_vals = ["linear", "rbf", "poly"]
SVM_error_val, values = svm_algorithm(X_train, y_train, X_val, y_val, vals, vals, kernel_vals)

print()
# Logistic Regression algorithm
Lambda_values = [0.001, 0.01, 0.1, 1, 3, 5, 10]
alpha = 0.5
num_iter = 10000
logreg_error_val, best_lambda = logreg_algorithm(X_train, y_train, X_val, y_val, alpha, num_iter, Lambda_values)

print("\nTesting the best model...")
if SVM_error_val < logreg_error_val:
    best_C, best_gamma, best_kernel = values
    svm_algorithm(np.append(X_train, X_val, axis=0), np.append(y_train, y_val, axis=0), X_test, y_test,
                  [best_C], [1/best_gamma], [best_kernel])
else:
    logreg_algorithm(np.append(X_train, X_val, axis=0), np.append(y_train, y_val, axis=0), X_test, y_test,
                     alpha, num_iter, [best_lambda])
