# ML hw2 problem 19.
import numpy as np
import random
import matplotlib.pyplot as plt

# parse data
with open('hw2_train.dat', 'r') as f:
	data = []
	for line in f:
		data.append( [float(x) for x in line.split()] )
	
DIMENSION = len(data[0])-1
SIZE = len(data)

Ein = [[0]*3] * DIMENSION # Ein, theta, s
for d in range(DIMENSION): # each dimension
	
	# d-th x and y, sorted by x
	x = sorted( [[row[d], row[9]] for row in data] )
	best_theta = best_s = 0
	min_err = SIZE
	for i in range(1, SIZE): # each data in one dimension
		theta = float((x[i][0] + x[i-1][0])/2)

		tmp_err_p = tmp_err_n = 0
		for k in range(SIZE):
			tmp_err_p += 1 if np.sign(x[k][0] - theta) != x[k][1] else 0
			tmp_err_n += 1 if (-1) * np.sign(x[k][0] - theta) != x[k][1] else 0
		
		if tmp_err_p < min_err:
			min_err = tmp_err_p
			best_theta = theta
			best_s = 1
		if tmp_err_n < min_err:
			min_err = tmp_err_n
			best_theta = theta
			best_s = -1
	
	Ein[d] = [(min_err/float(SIZE)), best_theta, best_s]
	print Ein[d]

dim = np.argmin([ Ein[d][0] for d in range(DIMENSION)])
print 'best dimension:', dim
print 'best Ein: ' '{:.3f}'.format(Ein[dim][0])
print 'best theta: ' '{:.3f}'.format(Ein[dim][1])
print 'best s: ' '{:3d}'.format(Ein[dim][2])
