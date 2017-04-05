import matplotlib.pyplot as plt
import numpy as np

plt.figure(1) # 创建图表1
plt.figure(2) # 创建图表2
ax1 = plt.subplot(211) # 在图表2中创建子图1
ax2 = plt.subplot(212) # 在图表2中创建子图2
 
x = np.linspace(0, 3, 100)
for i in range(5):
    plt.figure(1)  # 选择图表1
    plt.plot(x, np.exp(i*x/3))
    plt.sca(ax1)   # 选择图表2的子图1
    plt.plot(x, np.sin(i*x))
    plt.sca(ax2)   # 选择图表2的子图2
    plt.plot(x, np.cos(i*x))
 

a = [1, 2, 3, 4, 5]
b = [1, 5, 2, 4, 3]
plt.figure(3)
for i in range(5): # same as plt.plot(a, b, 'or')
	plt.plot(a[i], b[i], 'or')

plt.title("折線圖")
plt.xlabel("a")
plt.ylabel("b")
plt.show(3)
