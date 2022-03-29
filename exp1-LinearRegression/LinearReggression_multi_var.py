import numpy.matlib
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#代价函数
def costFunctionJ(x, y, theta) :
    #计算(x*theta - y) ^ 2
    inner = np.power(((x * theta.T) - y), 2)
    #计算  累加和/2m
    return np.sum(inner) / (2 * len(x))

#梯度下降函数
#theta为一维数组
#返回最后一个theta值，和迭代过程的代价J值
def gradientDescent(x, y, theta, alpha, iters):
    #theta缓存，用于迭代
    temp = np.matrix(np.zeros(theta.shape))

    #theta元素个数,为了计算theta0, theta1
    parameters = theta.ravel().shape[1]
    #每一次迭代的代价J值
    cost = np.zeros(iters)
    
    for i in range(iters):
        #计算结果与实际结果的差值
        error = (x * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))
            
        theta = temp
        cost[i] = costFunctionJ(x, y, theta)
        
    return theta, cost

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header = None, names = ['Size', 'Bedrooms', 'Price'])
data2.head()

#特征归一化处理
data2 = (data2 - data2.mean()) / data2.std()
data2.head()
#第0列插入1
data2.insert(0, 'Ones', 1)

#shape[1] --> 返回列数  
#shape[0] --> 返回行数
#shape      -> 返回(行，列)
cols = data2.shape[1]

#x -> 所有行，0到导数第二列
x = data2.iloc[:, 0: cols - 1]
#y -> 所有行，导数第二列至最后一列
y = data2.iloc[:, cols - 1: cols]

#将pandas.core.frame.DataFrame类型转成numpy.matrix类型 
x = np.matrix(x.values)
y = np.matrix(y.values)
#多变量，theta相对单变量要多一个元素
theta = np.matrix(np.zeros((1, 3)))
#代价函数，计算均方误差
print(costFunctionJ(x, y, theta))
#学习速率
alpha = 0.01
#迭代次数
iters = 1000
#批量梯度下降计算
g, cost = gradientDescent(x, y, theta, alpha, iters)
print(g)
print(costFunctionJ(x, y, g))

'''
#可视化
#创建x轴
#data的Population列，最小~最大，取100个
x = np.linspace(data.Population.min(), data.Population.max(), 100)
print(data.Population.max())
# y = theta_0 + theta_1 * x
f = g[0, 0] + (g[0, 1] * x)

#fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)
fig, (ax1, ax2) = plt.subplots(1, 2)
#画曲线
ax1.plot(x, f, 'r', label='Prediction')
#画散点
ax1.scatter(data.Population, data.Profit, label='Traning Data')
#图例的位置在第二象限
ax1.legend(loc=2)
ax1.set_xlabel('Population')
ax1.set_ylabel('Profit')
ax1.set_title('Predicted Profit vs. Population Size')
#plt.show()

#迭代过程代价J变化
ax2.plot(np.arange(iters), cost, 'r')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('Error vs. Training Epoch')
plt.show()

'''