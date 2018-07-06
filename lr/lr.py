# -*- coding: utf-8 -*-
# author: jsheng
# time: 18/7/1
# logistic regression

from numpy import *
import matplotlib.pyplot as plt
# 加载数据 ./lr/TestSet.txt
def loadDataset(file_name):
    dataMat = []
    labelMat = []

    fr = open(file_name)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(x):
    return 1.0/(1+exp(-x))

# 梯度上升算法
def gradAscent(dataMatin, dataLabel):
    dataMatrix = mat(dataMatin) #m*n
    print(shape(dataMatrix))
    labelMat = mat(dataLabel).transpose() #n*1

    m, n = shape(dataMatrix)
    print(m, n)
    weights = ones((n, 1))
    alpha = 0.0001
    for _ in range(5000):
        y = sigmoid(dataMatrix*weights)
        error = labelMat-y
        weights+=alpha*dataMatrix.transpose()*error

    return array(weights)


def plotBestFit(dataArr, labelMat, weights):
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)

    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X');plt.ylabel('Y')
    plt.show()

def train_test():
    dataMat, labelMat = loadDataset('data/lr/TestSet.txt')
    dataMat = array(dataMat)
    weights = gradAscent(dataMat, labelMat)

    plotBestFit(dataMat, labelMat, weights)

if __name__=='__main__':
    train_test()

