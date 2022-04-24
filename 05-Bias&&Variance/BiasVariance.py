import numpy as np
import scipy.io as sio

import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
#sns是一个对matplotlib二次封装的可视化库
import seaborn as sns

#线性回归代价函数
def cost(theta, x, y) :
    #计算(x*theta - y) ^ 2
    inner = np.power(((x.dot(theta.T)) - y), 2)
    #计算  累加和/2m
    return np.sum(inner) / (2 * len(x))

#线性回归正则化代价函数
def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + regularized_term



#线性回归梯度
def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)
    return inner / m
#线性回归正则化梯度
def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0  # don't regularize intercept theta
    regularized_term = (l / m) * regularized_term
    return gradient(theta, X, y) + regularized_term

#线性回归函数
def linear_regression_np(X, y, l=1):
    """linear regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res

def poly_features(x, power, as_ndarray=False):
    #创建x的1~3次项
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)
    return df.values if as_ndarray else df

#归一化
def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    #DataFrame.apply()传递一个函数，对DataFrame中每个元素进行处理
    #lambda返回一个函数，加入y= x + 1,则可表示为lambda x : x+1 
    return df.apply(lambda column: (column - column.mean()) / column.std())

#创建多项式特征
def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).values

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]
#绘制学习曲线
def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression_np(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)


def load_data():
    '''
    ex5data1.mat内容：
    X       : 12x1,训练集
    Xtest   : 21x1
    Xval    : 21x1,交叉验证集

    y       : 12x1,训练集
    ytest   : 21x1
    yval    : 21x1,交叉验证集
    '''
    d = sio.loadmat('ex5data1.mat')
    #map(function, iterable, ...)
    #使用function依次处理iterable，返回
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])

X, y, Xval, yval, Xtest, ytest = load_data()
df = pd.DataFrame({'water_level':X, 'flow':y})

#xLabel='water_level', yLabel='flow' data=df size=7
#fit_reg表示绘制x,y相关的回归模型
#sns.lmplot('water_level', 'flow', data=df, fit_reg=False, height=7)
#plt.show()
#把数据转成列的形式，并插入一列1
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

'''
theta = np.ones(X.shape[1])
J1 = cost(X, y, theta)
#print(J1)
G1 = gradient(theta, X, y)
#print(G1)
G2 = regularized_gradient(theta, X, y)
print(G2)
'''

final_theta = linear_regression_np(X, y, l=0).get('x')
b = final_theta[0] # intercept
m = final_theta[1] # slope
plt.scatter(X[:,1], y, label="Training data")
plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
plt.legend(loc=2)
plt.show()
################################################################
##以上都是以前的东西，普普通通的线性回归
################################################################

'''
training_cost, cv_cost = [], []
m = X.shape[0]
#不断增加训练集数量,进行线性回归计算
#使用得到的最优参数，分别计算训练集和交叉验证集代价值
for i in range(1, m+1):
    res = linear_regression_np(X[:i, :], y[:i], l=0)
    
    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)
    #print('tc={}, cv={}'.format(tc, cv))
    
    training_cost.append(tc)
    cv_cost.append(cv)
#绘制学习曲线
plt.plot(np.arange(1, m+1), training_cost, label='training cost')
plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
plt.legend(loc=1)
plt.show()      #根据学习曲线判断，有点欠拟合意思
'''

#因为上面的学习曲线显示的欠拟合/高偏差
#因此这里尝试增加多项式
X, y, Xval, yval, Xtest, ytest = load_data()
#扩展8阶多项式，归一化
X_poly, Xval_poly, Xtest_poly= prepare_poly_data(X, Xval, Xtest, power=8)
plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
plt.show()  #训练集代价为0，过拟合


#分别用训练集训练12个l，计算最优theta
#然后使用theta对交叉验证集计算代价值，找到代价之最小的那个l
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for l in l_candidate:
    res = linear_regression_np(X_poly, y, l)
    
    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)
    
    training_cost.append(tc)
    cv_cost.append(cv)
plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.legend(loc=2)
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()
#找到使交叉验证集代价值最小的那个l
print(l_candidate[np.argmin(cv_cost)])

'''
for l in l_candidate:
    theta = linear_regression_np(X_poly, y, l).x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xval_poly, yval)))
'''
