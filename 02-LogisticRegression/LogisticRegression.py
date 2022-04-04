import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.optimize as opt


#sigmoid函数
#参数可以是矩阵，或者实数
def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

#输出预测
def predict(theta, x):
    probability = sigmoid(x.dot(theta.T))
    return [1 if x >= 0.5 else 0 for x in probability]

# Logistic Regression代价函数
# 返回值[jVal, gradient] - 为了适应fmin_tcn调用
# 注意这个theta参数类型为数组
# 原因是fmin_tnc函数自动调用代价函数时，
# 会对自动将theta转换成array_like，因此参数theta的类型为array_like
def costFunctionJ(theta, x, y):
    #输入转换为矩阵形式
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    #H(x)
    hx= sigmoid(x.dot(theta.T)) 
    first = np.multiply(-y, np.log(hx))
    second = np.multiply((1 - y), np.log(1 - hx))

    #非矩阵形式
    '''
    parameters = theta.ravel().shape[1]
    grad = np.zeros(parameters)
    for i in range(parameters) :
        term = np.multiply(hx - y, x[:,i])
        grad[i] = ((1 / len(x)) * np.sum(term))
    '''
    #矩阵形式，下面两个都可以，只是grad的形状不一样
    #grad = 1.0/(len(X)) * (hx - y).T.dot(x)
    grad = 1.0/(len(x)) * x.T.dot(hx - y)
    return np.sum(first - second) / (len(x)), grad

#梯度下降
#其实和线性回归的梯度下降是一样的，只不过H(x)的实现不一样而已
#线性回归中，error = (x.dot(theta.T)) - y
#逻辑回归中，sigmoid(x.dot(theta.T)) - y
#工程中基本不会使用这个算法，而是使用现成的算法库
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
        error = sigmoid(x.dot(theta.T)) - y
        for j in range(parameters):
            #点乘
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))
            
        theta = temp
        cost[i] = costFunctionJ(x, y, theta)
    return theta, cost

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

#过滤'Admitted'为1的行
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

#显示点
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

#check sigmoid()
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

#x,y,theta初始化为numpy.array
#原因是后面的fmin_fnc调用代价函数时也会自动转成array_like
x = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
print(costFunctionJ(theta, x, y))

#梯度下降算法
#工程中基本不会使用这个算法，而是使用现成的算法库
#g, cost = gradientDescent(X, y, theta, 0.01, 1000)

#调用scipy库中的truncated newton (TNC) 实现寻找最优参数
#返回值[array, nfevel, rc]
# @array : 最优theta数组
# @nfevel: 函数评估的数量
# @rc : 返回码
'''
返回码定义：
-1 : Infeasible (lower bound > upper bound)
 0 : Local minimum reached (|pg| ~= 0)
 1 : Converged (|f_n-f_(n-1)| ~= 0)
 2 : Converged (|x_n-x_(n-1)| ~= 0)
 3 : Max. number of function evaluations reached
 4 : Linear search failed
 5 : All lower bounds are equal to the upper bounds
 6 : Unable to progress
 7 : User requested end of minimization 
 '''
result = opt.fmin_tnc(func = costFunctionJ, \
    x0 = theta, args=(x, y), messages = 0)
print(result)

#使用算法库返回的theta参数，验证代价函数
#应当比第一次的J(theta)值小很多
print(costFunctionJ(result[0], x, y))

theta_min = np.matrix(result[0])
#把数据集里的原始数组，都算一遍预测
predictions = predict(theta_min, x)
#预测正确的给1，不正确的给0
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
print(correct)
#计算正确的比例.89%，很理想
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))