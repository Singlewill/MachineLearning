import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#代价函数
def costFunctionJ(x, y, theta) :
    #计算(x*theta - y) ^ 2
    inner = np.power(((x * theta.T) - y), 2)
    #计算  累加和/2m
    return np.sum(inner) / (2 * len(x))

path = 'ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])
#第0列插入1
data.insert(0, 'Ones', 1)
#返回列数
cols = data.shape[1]
#x -> 所有行，0到导数第二列
x = data.iloc[:, 0: cols - 1]
#y -> 所有行，导数第二列至最后一列
y = data.iloc[:, cols - 1: cols]

#将pandas.core.frame.DataFrame类型转成numpy.matrix类型 
x = np.matrix(x.values)
y = np.matrix(y.values)

theta0 = np.linspace(-10, 10, 100)
theta1 = np.linspace(-1, 4, 100)

#生成网格矩阵，也就是一堆x、y坐标点，后面画图都得靠这个
#indexing指定了'ij'矩阵坐标系，默认是'xy'笛卡尔坐标系，xy反向的
g_x1, g_x2 = np.meshgrid(theta0, theta1, indexing = 'ij')


J_vals = np.zeros((theta0.shape[0] , theta1.shape[0]))
for i in range(0, theta0.shape[0]) :
    for j in range(0, theta1.shape[0]) :
        t = np.matrix([theta0[i], theta1[j]])
        J_vals[i, j] = costFunctionJ(x, y, t)


fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure = False)
fig.add_axes(ax)

p1 = ax.plot_surface(g_x1, g_x2, J_vals, rstride = 1, cstride = 1, cmap = cm.viridis)
plt.xlabel("theta0")
plt.ylabel("theta1")
plt.colorbar(p1)
plt.show()



#fig, ax = plt.subplots(1)
ax.set_xlim(-10, 10)
ax.set_ylim(-1, 4)
ax.set_zlim3d(0, 1000)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('J')
plt.figure(figsize=(5, 4), dpi=200)
plt.contour(g_x1, g_x2, J_vals, np.logspace(-2, 3, 20))
plt.show()

