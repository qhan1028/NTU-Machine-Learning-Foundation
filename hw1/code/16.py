# ML hw1 problem 16.
import numpy as np
import random
import matplotlib.pyplot as plt

ITER = 2000

# parse data
with open('hw1_15_train.dat', 'r') as f:
	data = []
	for line in f:
		x1, x2, x3, x4, y = [float(x) for x in line.split()]
		data.append([1.0, x1, x2, x3, x4, y])

# data length, error count of each iter
max_data = len(data)
data = np.array(data)
array_err_cnt = np.array( [0] * ITER)

np.set_printoptions( precision = 3, suppress = True)

# index
index = range(max_data)

# start training for 2000 times
for cnt_iter in range(ITER):
	random.shuffle(index)

	# one iteration of training
	data_err_cnt = np.array( [0] * max_data)
	w = np.array([0] * 5)
	cnt_err = finish = 0
	while not finish:
		error = 0
		for i in index:
			
			out_sign = np.sign( np.inner( w, np.array( data[i][0:5])))
			if out_sign != data[i][5]:
				error = 1
				w = w + data[i][0:5] * data[i][5]
				cnt_err += 1
				data_err_cnt[i] += 1
				
		if not error:
			finish = 1

	array_err_cnt[cnt_iter] = cnt_err

	# print result of one iteration
	print 'iter:', '{:4d}'.format(cnt_iter), ', w:', w, ', # updates:', cnt_err

ave_err = np.mean( array_err_cnt)

# print result
print 'Average of total updates: ', ave_err

# histogram
fig = plt.figure(1)

bins = range(80)
hist, bins = np.histogram(array_err_cnt, bins=bins)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)

plt.title( "16. # Updates vs. Frequency Histogram")
plt.xlabel( "# Updates")
plt.ylabel( "Frequency")
plt.show()

fig.savefig("16.png")
