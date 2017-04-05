# ML hw3.5 problem 13. ~ 20.
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab as pl

# file reader
def reader(filename):
	X, y = [], []
	with open(filename, 'r') as f:
		for line in f:
			data = line.split()
			X.append([float(i) for i in data[:-1]])
			y.append(int(data[-1]))

	return X, y

# ridge regression
def fit(X, y, Lambda):
	# dimension = len(X[0]), data_size = len(X)
	X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)
	y = np.array(y).reshape(len(y), 1)	
	
	# pinv: pseudo inverse
	w = np.dot( np.dot( np.linalg.inv( np.dot(X.T, X) + Lambda * np.eye(len(X[0])) ), X.T), y)
	return w

# prediction: vertical y_hat (yh)
def predict(w, X):
	X = np.append( np.ones((len(X), 1)), np.array(X), axis = 1) # add x0=1 in matrix X
	yh = (np.dot(X, w) > 0) * 2 - 1 # predict and convert to +1, -1 matrix
	return yh

# correct rate
def accuracy(y, yh):
	acc = np.sum( abs((y + yh) / 2)) / float(len(y))
	return acc

# 0/1 error
def error(w, X, y):
	y = np.array(y)
	yh = predict(w, X).reshape(len(y))
	return 1 - accuracy(y, yh)

LAMBDA = 1.126
X_all, y_all = reader('hw4_train.dat')
X_test, y_test = reader('hw4_test.dat')

# problem 13.
print('# problem 13.')
g = fit(X_all, y_all, LAMBDA)
Ein = error(g, X_all, y_all)
Eout = error(g, X_test, y_test)
print('  lambda = ', LAMBDA, ', Ein = ', Ein, ', Eout = ', Eout, sep='')

# problem 14. ~ 20. uses lambda = 2 ~ -10
LAMBDA = range(2, -11, -1)

# problem 14. 15.
print('\n# problem 14. 15.')
Ein = []
Eout = []
for power in LAMBDA:
	g = fit(X_all, y_all, 10**power)
	Ein.append(error(g, X_all, y_all))
	Eout.append(error(g, X_test, y_test))
	print('  log (lambda) = {:3d},'.format(power), 'Ein = {:.3f},'.format(Ein[-1]), 'Eout = {:.3f}'.format(Eout[-1]))

# figure 14.
fig = plt.figure(1)
plt.plot(LAMBDA, Ein, 'b')
plt.plot(LAMBDA, Ein, 'ro')
plt.title("14. Lambda and Ein")
plt.xlabel("Lambda")
plt.ylabel("Ein")
fig.savefig("14.png")

# figure 15.
fig = plt.figure(2)
plt.plot(LAMBDA, Eout, 'b')
plt.plot(LAMBDA, Eout, 'ro')
plt.title("15. Lambda and Eout")
plt.xlabel("Lambda")
plt.ylabel("Eout")
fig.savefig("15.png")

# problem 16. 17. 18.
print('\n# problem 16. 17. 18.')
X_train = X_all[0:120]
y_train = y_all[0:120]
X_val = X_all[120:200]
y_val = y_all[120:200]
Etrain = []
Eval = []
for power in LAMBDA:
	g = fit(X_train, y_train, 10**power)
	Etrain.append( error(g, X_train, y_train))
	Eval.append( error(g, X_val, y_val))
	Eout = error(g, X_test, y_test)
	print('  log (lambda) = {:3d},'.format(power), 'Etrain = {:.3f},'.format(Etrain[-1]), 'Eval = {:.3f},'.format(Eval[-1]), 'Eout = {:.3f}'.format(Eout))

# figure 16.
fig = plt.figure(3)
plt.plot(LAMBDA, Etrain, 'b')
plt.plot(LAMBDA, Etrain, 'ro')
plt.title("16. Lambda and Etrain")
plt.xlabel("Lambda")
plt.ylabel("Etrain")
fig.savefig("16.png")

# figure 17.
fig = plt.figure(4)
plt.plot(LAMBDA, Eval, 'b')
plt.plot(LAMBDA, Eval, 'ro')
plt.title("17. Lambda and Eval")
plt.xlabel("Lambda")
plt.ylabel("Eval")
fig.savefig("17.png")

# problem 19. 20.
Ecv_array = []
print('\n# problem 19. 20.')
for power in range(2, -11, -1):
	Etrain = Ecv = 0
	for div in range(0, 200, 40): # div ~ div+39 is D_val
		if div == 0:
			X_train = X_all[div+40 : 200]
			y_train = y_all[div+40 : 200]
		elif div == 160:
			X_train = X_all[0 : div]
			y_train = y_all[0 : div]
		else:
			X_train = np.append( X_all[0:div], X_all[div+40 : 200], axis=0)
			y_train = np.append( y_all[0:div], y_all[div+40 : 200], axis=0)
	
		X_val = X_all[div : div+40]
		y_val = y_all[div : div+40]

		g = fit(X_train, y_train, 10**power)
		Etrain += error(g, X_train, y_train) / 5
		Ecv += error(g, X_val, y_val) / 5

	Ecv_array.append(Ecv)
	print('  log (lambda) = {:3d},'.format(power), 'Etrain = {:.3f},'.format(Etrain), 'Ecv = {:.3f}'.format(Ecv))

# figure 19.
fig = plt.figure(5)
plt.plot(LAMBDA, Ecv_array, 'b')
plt.plot(LAMBDA, Ecv_array, 'ro')
plt.title("19. Lambda and Ecv")
plt.xlabel("Lambda")
plt.ylabel("Ecv")
fig.savefig("19.png")
