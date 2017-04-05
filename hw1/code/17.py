# ML hw1 problem 17.
import numpy as np
import random
import matplotlib.pyplot as plt

ITER = 2000
ETA = 0.25

np.set_printoptions( precision = 3, suppress = True)

# parse data
with open('hw1_15_train.dat', 'r') as f:
	data = []
	for line in f:
		x1, x2, x3, x4, y = [float(x) for x in line.split()]
		data.append([1.0, x1, x2, x3, x4, y])

# data length, error count of each iter
max_data = len(data)
data = np.array(data)
array_w_err_cnt = np.array([0] * ITER)
array_we_err_cnt = np.array([0] * ITER)

# index
index = range(max_data)

# start training for 2000 times
for cnt_iter in range(ITER):
	random.seed(cnt_iter)
	random.shuffle(index)

	# one iteration of training
	w = we = np.array([0] * 5)
	finish = w_cnt_err = we_cnt_err = 0
	while not finish:
		err_w = err_we = 0
		for i in index:
			adjust = data[i][0:5] * [data[i][5]]
		
			w_sign = np.sign(np.dot(w, data[i][0:5]))
			if w_sign != data[i][5]:
				err_w = 1
				w = w + adjust
				w_cnt_err += 1

			we_sign = np.sign(np.dot(we, data[i][0:5]))
			if we_sign != data[i][5]:
				err_we = 1
				we = we + adjust * ETA
				we_cnt_err += 1
				
		if not err_w and not err_we:
			finish = 1

	# sum up error of this iteration
	array_w_err_cnt[cnt_iter] = w_cnt_err
	array_we_err_cnt[cnt_iter] = we_cnt_err

	# print result of one iteration
	print 'iter:', '{:4d}'.format(cnt_iter), ', w: ', w, ', # updates:', w_cnt_err
	print 'iter:', '{:4d}'.format(cnt_iter), ', we:', we, ', # updates:', we_cnt_err

# result
w_ave_err = np.mean(array_w_err_cnt)
we_ave_err = np.mean(array_we_err_cnt)

# print result
print 'w: Average of total updates:', w_ave_err
print 'we: ETA:', ETA, ',Average of total updates:', we_ave_err

# w histogram
fig = plt.figure(1)
hist, bins = np.histogram(array_w_err_cnt, bins=80)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.title( "17-1. # Updates vs. Frequency Histogram (ETA = 1.0)")
plt.xlabel( "# Updates")
plt.ylabel( "Frequency")
plt.show()
fig.savefig("17_1.png")

# w histogram
fig = plt.figure(2)
hist, bins = np.histogram(array_we_err_cnt, bins=80)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.title( "17-2. # Updates vs. Frequency Histogram (ETA = 0.25)")
plt.xlabel( "# Updates")
plt.ylabel( "Frequency")
plt.show()

fig.savefig("17_2.png")
