import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.optimize as opt
#输出预测
def predict(theta, x):
    probability = sigmoid(x.dot(theta.T))
    return [1 if x >= 0.5 else 0 for x in probability]

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
def gradientReg(theta, x, y, lam=1):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    #实数*矩阵
    reg=(lam/len(x))*theta.T
    reg[0] = 0    # 第一项没有惩罚因子
    return gradient(theta, x, y) + reg

#代价函数
#相比于非正规化的逻辑回归代价函数，就是多了reg项
#其中reg项中的j是从1开始, 而不是0
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    #正规项中j从1开始
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg, gradientReg(theta, X, y)


path =  'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])


positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

#特征映射
#本来只有两个特征项，x1,x2，通过特征映射，扩展特征项，包含一系列高次项
degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']
data2.insert(3, 'Ones', 1)
for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.multiply(np.power(x1, i-j) ,np.power(x2, j))
#扩展完特征项后，将原始特征项丢弃
data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
#x1，x2特征映射后，是10项；补1，11项
theta2 = np.zeros(11)



learningRate = 1
#costReg(theta2, X2, y2, learningRate)

result2 = opt.fmin_tnc(func=costReg, x0=theta2, args=(X2, y2, learningRate), messages=0)
print(result2)

theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) / len(correct))
print ('accuracy = {0}%'.format(accuracy))