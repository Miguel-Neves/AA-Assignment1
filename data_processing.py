# Import relevant libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("cleveland.data", header=None)

y = data.values[:, 13]
y = y.reshape((len(y), 1))

# change the output to a binary classification
y = np.where(y > 0, 1, 0)  # presence (values 1,2,3,4); absence (value 0)
data = data.drop(columns=[13], axis=1)

# using One Hot Encoding for some columns (because they have only a few unique values)
data = pd.get_dummies(data, columns=[2, 6, 10, 11, 12])
# print(data.head())

X = data.values
# print(X.shape)
# print(y.shape)

# scaling the data to not overfit to the wrong features
sc = MinMaxScaler()
X_norm = sc.fit_transform(X)
# print(X_norm)

# split the data into 3 parts: training(60%), cross validation(20%) and test(20%)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.4)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# print(X_train.shape)
# print(X_test.shape)
# print(X_val.shape)

# save the data to files
data_train = pd.DataFrame(np.append(X_train, y_train, axis=1))
data_val = pd.DataFrame(np.append(X_val, y_val, axis=1))
data_test = pd.DataFrame(np.append(X_test, y_test, axis=1))
# print(data_train.shape)
# print(data_val.shape)
# print(data_test.shape)

data_train.to_csv("data_train.data", header=False, index=False)
data_val.to_csv("data_val.data", header=False, index=False)
data_test.to_csv("data_test.data", header=False, index=False)
