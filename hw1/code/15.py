# ML hw1 problem 15.
import numpy as np

np.set_printoptions( precision = 3, suppress = True)

with open('hw1_15_train.dat', 'r') as f:
	data = []
	for line in f:
		x1, x2, x3, x4, y = [float(x) for x in line.split()]
		data.append([1.0, x1, x2, x3, x4, y])

# max data length
max_data = len(data)
data_err_cnt = np.array([0] * max_data)

data = np.array(data)
w = np.array([0] * 5)

cnt_err = 0
while 1:
	error = 0
	for i in range(max_data):
		out_sign = np.sign( np.dot( w, data[i][0:5]))	
		if out_sign != data[i][5]:
			error = 1
			w = w + data[i][0:5] * data[i][5];
			cnt_err += 1
			data_err_cnt[i] += 1
			
	if not error:
		break

print 'Final w:'
idx = np.argmax( data_err_cnt)
print 'Data with max updates: index =', idx, ', error count =', data_err_cnt[idx]
print 'Total updates: ', cnt_err
