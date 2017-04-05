# ML problem 17. 18.
import numpy as np
import random
import matplotlib.pyplot as plt

ITER = 5000
SIZE = 10

np.set_printoptions(precision=3)

Ein = np.array([0.0]*ITER)
Eout = np.array([0.0]*ITER)

for it in range(ITER):

	np.random.seed(it)
	data = np.random.uniform(-1, 1, SIZE) # generate data
	data = np.append(data, [1, -1])	# first and last are -1 and 1
	data = np.sort(data)
	print it, data[1:11]

	# 0.2 possibility of error
	y = np.sign( np.multiply(np.sign(data[1:11]), np.random.uniform(-0.2, 0.8, SIZE)))
	print y

	best_theta = best_s = 0
	min_err = 10.0
	for j in range(SIZE + 1):	# compute 11 thetas

		theta = 0.5 * (data[j] + data[j+1])
		#print 'theta', j, '=', '{:.3f}'.format(theta)

		hy = np.array([0] * SIZE) # hypothesis y
		for k in range(SIZE):
			hy[k] = np.sign(data[k+1] - theta)

		tmp_err = 0
		for k in range(SIZE):	# s = +1
			tmp_err += 1 if hy[k] != y[k] else 0

		if tmp_err <= min_err:
			min_err = tmp_err
			best_theta = theta
			best_s = 1
		#print 's = +1, Ein =', tmp_err

		tmp_err = 0
		for k in range(SIZE):	# s = -1
			tmp_err += 1 if (-1) * hy[k] != y[k] else 0

		if tmp_err <= min_err:
			min_err = tmp_err
			best_theta = theta
			best_s = -1
		#print 's = -1, Ein =', tmp_err
		
	Eout[it] = 0.5 + 0.3 * best_s * (abs(best_theta) - 1)
	Ein[it] = min_err / float(SIZE)
	print 'best: theta =', '{:.3f}'.format(best_theta), 's =', best_s
	print 'Ein =', '{:.3f}'.format(Ein[it]), 'Eout =', '{:.3f}'.format(Eout[it])
	print

ave_Ein = np.mean(Ein)
ave_Eout = np.mean(Eout)

print 'average Ein =', '{:.3f}'.format(ave_Ein), 'average Eout =', '{:.3f}'.format(ave_Eout)

# histogram
plt.figure()
n, bins, patches = plt.hist(Ein, [ -0.05 + 0.1 * i for i in range(12) ], normed = False, histtype = "bar", rwidth = 0.8)
plt.title("17. Ein distribution")
plt.xlabel("Ein")
plt.ylabel("Frequency")
plt.savefig("17.png")
plt.show()

plt.figure()
n, bins, patches = plt.hist(Eout, [ -0.05 + 0.05 * i for i in range(23) ], normed = False, histtype = "bar", rwidth = 0.8)
plt.title("18. Eout distribution")
plt.xlabel("Eout")
plt.ylabel("Frequency")
plt.savefig("18.png")
plt.show()
