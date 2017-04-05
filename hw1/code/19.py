# ML hw1 problem 19.
import numpy as np
import random
import matplotlib.pyplot as plt

ITER = 2000
POCKET_UPDATE = 100

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

# train data length, numpy
max_data = len(data)
data = np.array(data)
array_err_rate = np.array( [0.] * ITER)

# test data length, numpy
max_test_data = len(test_data)
test_data = np.array(test_data)
array_test_err_rate = np.array( [0.] * ITER)

# index
index = range(max_data)

# start training for ITER times
for cnt_iter in range(ITER):
	random.seed(cnt_iter)
	random.shuffle(index)

	# one iteration of training
	w = wh = np.array([0.] * 5)
	need_update = POCKET_UPDATE
	while need_update:
		# find a new w that corrected with one mistake
		mistake_list = []
		for i in index:
			if np.sign( np.dot( w, data[i][0:5])) != data[i][5]:
				mistake_list.append(i)
		
		i = random.choice(mistake_list)
		w = w + data[i][0:5] * data[i][5]

		# compare wh and w
		wh_err_cnt = w_err_cnt = 0
		for i in index:	
			if np.sign( np.dot( w, data[i][0:5])) != data[i][5]:
				w_err_cnt += 1
		
			if np.sign( np.dot( wh, data[i][0:5])) != data[i][5]:
				wh_err_cnt += 1

		if w_err_cnt < wh_err_cnt:
			wh = w
			best_err_cnt = w_err_cnt
		else:
			w = wh
	
		need_update -= 1
	
	# in error rate of this iteration
	array_err_rate[cnt_iter] = float(best_err_cnt)/float(max_data)

	test_error = 0
	for j in range(max_test_data):
		test_sign = np.sign( np.dot( wh, test_data[j][0:5]))
		if test_sign != test_data[j][5]:
			test_error += 1
	
	array_test_err_rate[cnt_iter] = float(test_error)/float(max_test_data)
	print 'iter:', '{:4d}'.format(cnt_iter), ' wh:', wh, ' Ein rate:', '{:.3f}'.format(array_err_rate[cnt_iter]), ' Eout rate:', '{:.3f}'.format(array_test_err_rate[cnt_iter])
	
# average in error rate
ave_err_rate = np.mean(array_err_rate)
ave_test_err_rate = np.mean(array_test_err_rate)

# print result
print 'Average in error rate:', ave_err_rate
print 'Average out error rate:', ave_test_err_rate

# histogram
hist, bins = np.histogram(array_test_err_rate, bins=100)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.title( "19. Out Error Rate vs. Frequency Histogram")
plt.xlabel( "Error Rate")
plt.ylabel( "Frequency")
plt.show()
