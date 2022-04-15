import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.preprocessing import OneHotEncoder
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#前向传播函数
#3层神经网络,因此只有theta1和theta2
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h

#3层神经网络，代价函数
# @params           : 输入theta一维化结果
# @input_size       : 输入层神经元数量
# @hidden_size      : 隐藏神经元数量
# @num_labels       : 输出层单元数量
# @learning_rate    : 学习速率
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    #theta1大小为(隐藏层单元数 x 输入层单元数+1)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    #theta1大小为(输出单元数 x 隐藏单元数+1)
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    #前向传播算法
    #等于是把5000个样本的y=h(x)值都算了一遍
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # compute the cost
    J = 0
    #逻辑回归的代价函数公式
    #J(θ)= -y*log(h(x))-(1-y)log(1-h(x))
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    return J

#3层神经网络，代价函数(正则化)
# @params           : 输入theta一维化结果
# @input_size       : 输入层神经元数量
# @hidden_size      : 隐藏神经元数量
# @num_labels       : 输出层单元数量
# @learning_rate    : 学习速率
def costReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    # add the cost regularization term
    #所有theta平方的和 / learning_rate / 2m
    #theta从j=1开始
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    return J


data = loadmat('ex4data1.mat')
X = data['X']   #5000x400
y = data['y']   #5000x1
encoder = OneHotEncoder(sparse = False)
#将y数据中的1~10转成独热码形式
#2 -> [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
y_onehot = encoder.fit_transform(y) #5000x10

# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# 随机初始化完整网络参数大小的参数数组
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

c1 = cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(c1)
c2 = costReg(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(c2)