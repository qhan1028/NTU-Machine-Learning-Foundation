# ML hw1 problem 20.
import numpy as np
import random
import matplotlib.pyplot as plt

ITER = 2000
UPDATE = 100

# set print options
np.set_printoptions( precision = 3, suppress = True)

# parse data
with open('hw1_18_train.dat', 'r') as f:
	data = []
	for line in f:
		x1, x2, x3, x4, y = [float(x) for x in line.split()]
		data.append([1.0, x1, x2, x3, x4, y])

with open('hw1_18_test.dat', 'r') as test_f:
	test_data = []
	for line in test_f:
		x1, x2, x3, x4, y = [float(x) for x in line.split()]
		test_data.append([1.0, x1, x2, x3, x4, y])

# train data length, error count of each iter
max_data = len(data)
data = np.array(data)

# test data length
max_test_data = len(test_data)
test_data = np.array(test_data)
array_test_err_rate = np.array( [0.] * ITER)

# index
index = range(max_data)

# start training for ITER times
for cnt_iter in range ( 0, ITER, 1):
	random.seed(cnt_iter)
	random.shuffle(index)

	# one iteration of training
	w = np.array([0.] * 5)
	need_update = UPDATE
	while 1:
		# find a new w that corrected with one mistake
		for i in index:
			sign = np.sign( np.inner( w, data[i][0:5]))
			if sign > 0 and data[i][5] <= 0:
				w = w + data[i][0:5] * data[i][5]
				need_update -= 1
			if sign <= 0 and data[i][5] > 0:
				w = w + data[i][0:5] * data[i][5]
				need_update -= 1
		if need_update <= 0: 
			break
	
	# out error rate of this iteration
	test_error = 0
	for j in range(max_test_data):
		test_sign = np.sign( np.inner( w, test_data[j][0:5]))
		if test_sign > 0 and test_data[j][5] <= 0:
			test_error += 1
		if test_sign <= 0 and test_data[j][5] > 0:
			test_error += 1
	
	array_test_err_rate[cnt_iter] = float(test_error)/float(max_test_data)
	print 'iter:', '{:4d}'.format(cnt_iter), ' w:', w, ' Eout rate:', '{:.3f}'.format(array_test_err_rate[cnt_iter])
	
# average in error rate
ave_test_err_rate = np.mean(array_test_err_rate)

# print result
print 'Average out error rate:', ave_test_err_rate

# histogram
fig = plt.figure(1)

hist, bins = np.histogram(array_test_err_rate, bins=100)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.title( "20. Out Error Rate vs. Frequency Histogram")
plt.xlabel( "Error Rate")
plt.ylabel( "Frequency")
plt.show()

fig.savefig("20.png")
