import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
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

# a * (1 - a), 
# a = sigmoid(z)
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

#3层神经网络，反向传播
# @params           : 输入theta一维化结果
# @input_size       : 输入层神经元数量
# @hidden_size      : 隐藏神经元数量
# @num_labels       : 输出层单元数量
# @X,y              : 输入学习样本
# @learning_rate    : 学习速率
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # initializations
    #theta1大小为(隐藏层单元数 x 输入层单元数+1)
    #theta1大小为(输出单元数 x 隐藏单元数+1)
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    
    # perform backpropagation
    for t in range(m):  # m=400
        #第一层输入（加偏置）
        a1t = a1[t,:]  # (1, 401)
        #第二层加权输入
        z2t = z2[t,:]  # (1, 25)
        #第二层输出
        a2t = a2[t,:]  # (1, 26)
        #第三层输出
        ht = h[t,:]  # (1, 10)
        #实际输出
        yt = y[t,:]  # (1, 10)
        
        #误差单元，也是小delta3
        d3t = ht - yt  # (1, 10)
        
        #隐藏层加偏置
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        #误差单元，小delta-2
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        
        #代价函数偏导数，大delta-1 = a1 * 小delta-2
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        #大delta-2 = a2 * 小delta-3
        delta2 = delta2 + d3t.T * a2t
        
    #梯度 = 偏导数/len(X)
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad

#3层神经网络，反向传播(正则化)
# @params           : 输入theta一维化结果
# @input_size       : 输入层神经元数量
# @hidden_size      : 隐藏神经元数量
# @num_labels       : 输出层单元数量
# @X,y              : 输入学习样本
# @learning_rate    : 学习速率
def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # initializations
    #theta1大小为(隐藏层单元数 x 输入层单元数+1)
    #theta1大小为(输出单元数 x 隐藏单元数+1)
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    
    # add the cost regularization term
    #正则化代价函数
    #加上theta的平方和项
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    # perform backpropagation
    for t in range(m):  # m=400
        #第一层输入（加偏置）
        a1t = a1[t,:]  # (1, 401)
        #第二层加权输入
        z2t = z2[t,:]  # (1, 25)
        #第二层输出
        a2t = a2[t,:]  # (1, 26)
        #第三层输出
        ht = h[t,:]  # (1, 10)
        #实际输出
        yt = y[t,:]  # (1, 10)
        
        #误差单元，也是小delta3
        d3t = ht - yt  # (1, 10)
        
        #隐藏层加偏置
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        #误差单元，小delta-2
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        
        #代价函数偏导数，大delta-1 = a1 * 小delta-2
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        #大delta-2 = a2 * 小delta-3
        delta2 = delta2 + d3t.T * a2t
        
    #梯度 = 偏导数/len(X)
    delta1 = delta1 / m
    delta2 = delta2 / m

    # 梯度正则化，加上(theta + learning_rate) / m
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate)  / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad

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

#不带正则化的代价函数
c1 = cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(c1)
#带正则化的
c2 = costReg(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(c2)

#向后传播，J带正则，grad不带正则
J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(J)

#向后传播，带正则
J2, grad = backpropReg(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(J2)


#目标函数也不一定收敛，这里指定迭代次数250
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), 
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)


#利用高级优化函数返回的theta值进行预测
#fmin.x是返回的theta值
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

print(h.shape)
#argmax(h, axis=1)返回每一行的最大值的索引
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))