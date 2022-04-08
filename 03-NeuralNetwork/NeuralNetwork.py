import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from scipy.optimize import minimize

#sigmoid函数
#参数可以是矩阵，或者实数
def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def gradient(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    hx= sigmoid(x.dot(theta.T)) 
    return 1.0 / (len(x)) * (x.T.dot(hx - y))
#这里要保证参数是矩阵，最好还是在函数内部进行一下处理
def gradientReg(theta, x, y, learningRate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    #实数*矩阵
    reg=(learningRate / len(x)) * theta.T
    reg[0] = 0    # 第一项没有惩罚因子
    return gradient(theta, x, y) + reg

#代价函数
#相比于非正规化的逻辑回归代价函数，就是多了reg项
#其中reg项中的j是从1开始, 而不是0
#这里的代价函数不再返回grad
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    #正规项中j从1开始
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg, gradientReg(theta, X, y, learningRate)
    #return np.sum(first - second) / len(X) + reg

def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        #每一个oneVSall计算，把这个one的结果挑出来置1，其他置0
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        
        # minimize the objective function
        #fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradientReg)
        fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=True)
        all_theta[i-1,:] = fmin.x
    
    return all_theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # convert to matrices
    X = np.matrix(X)        #5000 x 401
    all_theta = np.matrix(all_theta)
    

    #all_theta每一行代表一个y值的最佳theta值
    # h.shape = 5000 x 10
    #每一行的10个值，是10个最佳theta的运算结果
    h = sigmoid(X * all_theta.T)

    
    #print(h.shape)  #5000x10
    #h的每一行的最大值的索引
    #这里有个巧合，索引值 + 1 =数据值
    h_argmax = np.argmax(h, axis=1)
    #print(h_argmax.shape) #5000x1
    
    #h_argmax中保存的是索引，0~9
    #y只保存的是具体的值，1~10,
    # 后面为了比较相等，因此这里需要加1
    h_argmax = h_argmax + 1
    
    return h_argmax

#ex3data1.mat是matlab的文件格式 
# #可以在matlab中 load ex3data1.mat查看变量具体内容
# 文件中包含了5000组数据，每组输入，20x20 = 400灰度像素，
# 每组输出描述了数字识别结果：0~9
# 即：X=5000x400, y = 5000x1
data = loadmat('ex3data1.mat')
#print(data['X'].shape)  #5000x400
#print(data['y'].shape)  #5000x1
print(data['y'][0])

#all_theta = 10 * 401
#每一行表示了对应的y值的最佳theta
all_theta = one_vs_all(data['X'], data['y'], 10, 1)
print(all_theta.shape)

y_pred = predict_all(data['X'], all_theta)
#这里有个巧合，索引值 =数据值
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
