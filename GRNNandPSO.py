import numpy as np
from pyswarm import pso
from sklearn.metrics import mean_squared_error


def distance(X,Y):
    '''计算两个样本之间的距离
    '''
    return np.sqrt(np.sum(np.square(X-Y),axis = 0)).reshape(1,1)

def distance_mat(trainX,testX):
    '''计算待测试样本与所有训练样本的欧式距离
    input:trainX(mat):训练样本
          testX(mat):测试样本
    output:Euclidean_D(mat):测试样本与训练样本的距离矩阵
    '''
    m,n = np.shape(trainX)
    p = np.shape(testX)[0]
    Euclidean_D = np.mat(np.zeros((p,m)))
    for i in range(p):
        for j in range(m):
            Euclidean_D[i,j] = distance(testX[i,:],trainX[j,:])[0,0]
    return Euclidean_D

def Gauss(Euclidean_D,sigma):
    '''测试样本与训练样本的距离矩阵对应的Gauss矩阵
    input:Euclidean_D(mat):测试样本与训练样本的距离矩阵
          sigma(float):Gauss函数的标准差
    output:Gauss(mat):Gauss矩阵
    '''
    m,n = np.shape(Euclidean_D)
    Gauss = np.mat(np.zeros((m,n)))
    for i in range(m):
        for j in range(n):
            Gauss[i,j] = np.exp(- Euclidean_D[i,j] / (2 * (sigma ** 2)))
    return Gauss

def sum_layer(Gauss,trY):
    '''求和层矩阵，列数等于输出向量维度+1,其中0列为每个测试样本Gauss数值之和
    '''
    m,l = np.shape(Gauss)
    n = np.shape(trY)[1]
    sum_mat = np.mat(np.zeros((m,n+1)))
    ## 对所有模式层神经元输出进行算术求和
    for i in range(m):
        sum_mat[i,0] = np.sum(Gauss[i,:],axis = 1) ##sum_mat的第0列为每个测试样本Gauss数值之和
    ## 对所有模式层神经元进行加权求和
    for i in range(m):
        for j in range(n):
            total = 0.0
            for s in range(l):
                total += Gauss[i,s] * trY[s,j]
            sum_mat[i,j+1] = total           ##sum_mat的后面的列为每个测试样本Gauss加权之和
    return sum_mat

def output_layer(sum_mat):
    '''输出层输出
    input:sum_mat(mat):求和层输出矩阵
    output:output_mat(mat):输出层输出矩阵
    '''
    m,n = np.shape(sum_mat)
    output_mat = np.mat(np.zeros((m,n-1)))
    for i in range(n-1):
        output_mat[:,i] = sum_mat[:,i+1] / sum_mat[:,0]
    return output_mat

def fitness(sigma, X_train, X_test, y_train, y_test):
    Euclidean_D = distance_mat(X_train, X_test)
    Gauss_mat = Gauss(Euclidean_D, sigma)
    sum_mat = sum_layer(Gauss_mat, y_train)
    y_pred = output_layer(sum_mat)
    return np.sqrt(mean_squared_error(y_test, y_pred))

# 定义粒子群优化函数
def optimize_sigma(X_train, X_test, y_train, y_test):
    # 粒子群优化的目标函数
    def objective(sigma):
        return fitness(sigma, X_train, X_test, y_train, y_test)

    # 设置优化的上下界
    lb =[0.1]
    ub =[1.0]

    # 使用粒子群优化算法寻找最优解
    best_sigma, _= pso(objective, lb, ub)

    return best_sigma
