# ML hw3.5 problem 11. 12.
import numpy as np
import math as m

# file reader
def reader(filename):
	X, y = [], []
	with open(filename, 'r') as f:
		for line in f:
			data = line.split()
			X.append([float(i) for i in data[:-1]])
			y.append(int(data[-1]))

	return X, y

# logistic regression
def fit(X, y, T, eta, SGD):
	# dimension = len(X[0]), data_size = len(X)
	X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)
	y = np.array(y).reshape(len(y), 1)
	w = np.zeros((len(X[0]), 1))
	
	for t in range(T):
		if SGD:
			i = t % len(X)
			Grad_Ein = (theta(-y[i] * np.dot(X[i], w)) * (-y[i] * X[i])).reshape(len(X[0]), 1)

		else:
			Grad_Ein = np.sum( theta(-y * np.dot(X, w)) * (-y * X), axis=0).reshape(len(X[0]), 1) / len(X)
		
		w = w - eta * Grad_Ein

	return w

# logistic function
def theta(s):
	return 1.0 / (1.0 + np.exp(-s))

# prediction: vertical yh
def predict(w, X):
	X = np.append( np.ones((len(X), 1)), np.array(X), axis = 1) # add x0 = 1 in X
	return np.rint( theta(np.dot(X, w))) * 2 - 1 # predict and convert to +1, -1 matrix

# correct rate
def accuracy(y, yh):
	return np.sum( abs((y + yh) / 2)) / float(len(y))

# 0/1 error
def error(w, X, y):
	y = np.array(y)
	yh = predict(w, X).reshape(len(y))
	return 1 - accuracy(y, yh)

SGD = 1
ETA = 0.001
T = 2000

# problem 11.
X_train, y_train = reader('hw3_train.dat')
X_test, y_test = reader('hw3_test.dat')
g = fit(X_train, y_train, T, ETA, not SGD)
err = error(g, X_test, y_test)
print('# Problem 11.\n  Weight Vector\n', g)
print('  Eout = ', err, '\n')

# problem 12.
X_train, y_train = reader('hw3_train.dat')
X_test, y_test = reader('hw3_test.dat')
g = fit(X_train, y_train, T, ETA, SGD)
err = error(g, X_test, y_test)
print('# Problem 12.\n  Weight Vector\n', g)
print('  Eout = ', err)
