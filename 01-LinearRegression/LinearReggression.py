import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 正规方程
def normalEqn(X, y):
    #X.T@X等价于X.T.dot(X)
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

#代价函数
def costFunctionJ(x, y, theta) :
    #计算(x*theta - y) ^ 2
    inner = np.power(((x.dot(theta.T)) - y), 2)
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
        #这里是矩阵乘法
        error = (x.dot(theta.T)) - y
        for j in range(parameters):
            #点乘
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))
            
        theta = temp
        cost[i] = costFunctionJ(x, y, theta)
        
    return theta, cost

path = 'ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])
#data.head()
#data.describe()

#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#plt.show()画图
#plt.show() 
#第0列插入1
data.insert(0, 'Ones', 1)

#shape[1] --> 返回列数  
#shape[0] --> 返回行数
#shape      -> 返回(行，列)
cols = data.shape[1]

#x -> 所有行，0到导数第二列
x = data.iloc[:, 0: cols - 1]
#y -> 所有行，导数第二列至最后一列
y = data.iloc[:, cols - 1: cols]

#将pandas.core.frame.DataFrame类型转成numpy.matrix类型 
x = np.matrix(x.values)
y = np.matrix(y.values)
#因为x是97x2,为了能够相乘，theta必须得2x1
#对于1维矩阵，习惯初始化为1xn的形式
#这里初始化为1x2，在函数内部会做转置
#对于多变量线性回归，这里增加theta的长度即可
theta = np.matrix(np.zeros((1, 2)))
print('初始代价函数Jval值:')
#代价函数，计算均方误差
print(costFunctionJ(x, y, theta))
#学习速率
alpha = 0.01
#迭代次数
iters = 1000
#批量梯度下降计算
g, cost = gradientDescent(x, y, theta, alpha, iters)
print('梯度下降后的Theta值:')
print(g)


#正规方程
g2 = normalEqn(x, y)
print('正规方程后的Theta值:')
print(g2)

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