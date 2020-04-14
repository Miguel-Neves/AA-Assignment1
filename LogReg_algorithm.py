import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


def sigmoid(z):
    """
    return the sigmoid of z
    """
    gz = 1 / (1 + np.exp(-z))
    return gz


def costFunctionReg(X, y, theta, Lambda):
    """
    Take in numpy array of  data X, labels y and theta, to return the regularized cost function and gradients
    of the logistic regression classifier
    """
    # number of training examples
    m = len(y)
    # vector of the model predictions for all training examples
    h = sigmoid(np.dot(X, theta))
    error = (-y * np.log(h)) - ((1 - y) * np.log(1 - h))
    # cost function without regularization term
    cost = sum(error) / m
    # add regularization term to the cost function
    regCost = cost + Lambda / (2 * m) * sum(theta[1:] ** 2)
    # gradient of theta_0
    grad_0 = (1 / m) * np.dot(X.transpose(), (h - y))[0]
    # vector of gradients of theta_j from j=1:n (adding the regularization term of the gradient)
    grad = (1 / m) * np.dot(X.transpose(), (h - y))[1:] + (Lambda / m) * theta[1:]
    # all gradients in a column vector shape
    grad_all = np.append(grad_0, grad)
    grad_all = grad_all.reshape((len(grad_all), 1))
    return regCost[0], grad_all


def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps with learning rate of alpha
    return theta and the list of the cost of theta during each iteration
    """
    J_history = []
    for i in range(num_iters):
        cost, grad = costFunctionReg(X, y, theta, Lambda)
        theta = theta - alpha * grad
        J_history.append(cost)
    return theta, J_history


def validationCurve(Xtrain, ytrain, Xval, yval, learn_rate, num_iter, Lambda_array):
    """
    Returns the best lambda and the respective train and cross validation set errors
    """
    m = len(ytrain)  # Number of training examples
    n = len(Xtrain[0])  # number of features
    mval = len(yval)  # Number of validation examples
    error_train, error_val = [], []
    for lam in Lambda_array:
        theta_ini = np.zeros((n, 1))
        theta = gradientDescent(Xtrain, ytrain, theta_ini, learn_rate, num_iter, lam)[0]
        pred_train = np.where(np.dot(Xtrain, theta) > 0, 1, 0)
        pred_val = np.where(np.dot(Xval, theta) > 0, 1, 0)
        error_train.append(1 / (2 * m) * np.sum((pred_train - ytrain) ** 2))
        error_val.append(1 / (2 * mval) * np.sum((pred_val - yval) ** 2))
    return error_train, error_val


def logreg_algorithm(X_train, y_train, X_val, y_val, alpha, num_iter, Lambda_values):
    y_train = y_train.reshape((len(y_train), 1))
    y_val = y_val.reshape((len(y_val), 1))
    initial_theta = np.zeros((X_train.shape[1], 1))

    # calculate and plot the cost function trajectory
    """
    for l in Lambda_values:
        J_history = gradientDescent(X_train, y_train, initial_theta, alpha, num_iter, l)[1]
        plt.plot(J_history)
        # print("Lambda:", l, "Cost:", round(J_history[-1], 3))
    plt.xlabel("Iteration")
    plt.ylabel("$j(\Theta)$")
    plt.legend(Lambda_values)
    plt.show()
    """

    error_train, error_val = validationCurve(X_train, y_train, X_val, y_val, alpha, num_iter, Lambda_values)
    plt.plot(Lambda_values, error_train, label="Train")
    plt.plot(Lambda_values, error_val, label="Cross Validation", color="r")
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    print(" - Logistic Regression Algorithm (regularized) - ")
    print("Learning rate:", alpha)
    print("Number of iterations:", num_iter)
    print("Lambda values:", Lambda_values)
    print("\tResults")

    index = np.argmin(error_val).min(initial=0)
    best_lambda = Lambda_values[index]
    best_error_val = error_val[index]
    print("Best Lambda:", best_lambda)
    print("Error:", best_error_val)

    # Calculate accuracy, precision, sensitivity and f1 score for the best lambda value
    theta = gradientDescent(X_train, y_train, initial_theta, alpha, num_iter, best_lambda)[0]
    pred_val = np.where(np.dot(X_val, theta) > 0, 1, 0)
    print("Accuracy:", round(metrics.accuracy_score(y_val, pred_val)*100, 1), "%")
    print("Precision:", round(metrics.precision_score(y_val, pred_val)*100, 1), "%")
    print("Sensitivity (recall):", round(metrics.recall_score(y_val, pred_val)*100, 1), "%")
    print("F1 Score:", round(metrics.f1_score(y_val, pred_val)*100, 1), "%")
    print("Confusion matrix:\n", metrics.confusion_matrix(y_val, pred_val))
    # TN FP
    # FN TP

    return best_error_val, best_lambda
