import numpy as np
from sklearn import metrics
from sklearn.svm import SVC


def validation_curve(Xtrain, ytrain, Xval, yval, C_vals, sigma_vals, kernel):

    values, error_train, error_val = [], [], []
    m_train = Xtrain.shape[0]
    m_val = Xval.shape[0]

    for i in C_vals:
        C = i
        for j in sigma_vals:
            gamma = 1 / j
            for k in kernel:
                classifier = SVC(C=C, gamma=gamma, kernel=k, coef0=10.0)
                classifier.fit(Xtrain, ytrain)

                # predicting the output for training and validation sets
                pred_train = classifier.predict(Xtrain)
                pred_val = classifier.predict(Xval)

                # computing the error
                error_train.append(1/(2*m_train) * np.sum((pred_train - ytrain)**2))
                error_val.append(1/(2 * m_val) * np.sum((pred_val - yval) ** 2))
                values.append((i, gamma, k))  # (C, gamma, kernel)
                # print("(C, gamma, kernel) = ", (i,gamma,k))
                # print("Error val: ", 1/(2 * m_val) * np.sum((pred_val - yval) ** 2))

    return values, error_train, error_val


def svm_algorithm(X_train, y_train, X_val, y_val, C_vals, sigma_vals, kernel_vals):
    print(" - SVM algorithm - ")
    print("C values:", C_vals)
    print("sigma values:", sigma_vals)
    print("kernel values:", kernel_vals)

    values, error_train, error_val = validation_curve(X_train, y_train.ravel(), X_val, y_val.ravel(), C_vals, sigma_vals, kernel_vals)

    index = np.argmin(error_val).min()
    best_C, best_gamma, best_kernel = values[index]

    print("\tResults")
    print("Best values: C=", best_C, " sigma=", 1/best_gamma, " kernel=", best_kernel)
    print("Error: ", error_val[index])

    classifier = SVC(C=best_C, gamma=best_gamma, kernel=best_kernel, coef0=10.0)
    classifier.fit(X_train, y_train)

    pred_val = classifier.predict(X_val)
    print("Accuracy:", round(metrics.accuracy_score(y_val, pred_val)*100, 1), "%")
    print("Precision:", round(metrics.precision_score(y_val, pred_val)*100, 1), "%")
    print("Sensitivity (recall):", round(metrics.recall_score(y_val, pred_val)*100, 1), "%")
    print("F1 score:", round(metrics.f1_score(y_val, pred_val)*100, 1), "%")
    return error_val[index], values[index]
