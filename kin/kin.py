# -*- coding: utf-8 -*-
# author: jsheng
# time: 18/7/1

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator
from os import listdir
from collections import Counter

# 数据格式
# 26052	1.441871	0.805124	1
# 75136	13.147394	0.428964	1
# 38344	1.669788	0.134296	1

# 处理数据
def file2matrix(filename):
    fr = open(filename)
    numfileline = len(fr.readlines())

    reMatrix = np.zeros((numfileline, 3))  # 数据
    labelVec = []  # 标签
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        linelist = line.split('\t')
        reMatrix[index, :] = linelist[0:3]
        labelVec.append(int(linelist[-1]))
        index += 1

    return reMatrix, labelVec

def autoNorm(dataset):
    # 归一化数据
    minVals = dataset.min(0)
    maxVals = dataset.max(0)

    ranges = maxVals-minVals
    normDataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    # b = tile(a, (m, n)):即是把a数组里面的元素复制n次放进一个数组c中，然后再把数组c复制m次放进一个数组b中
    normDataset = dataset - np.tile(minVals, (m, 1))
    normDataset = normDataset/np.tile(ranges, (m, 1))

    return normDataset, ranges, minVals

def classify(inX, dataset, labels, k):
    # 计算距离，算最近的k近邻
    '''
    :param inX: 输入向量
    :param dataset: test
    :param labels: 标签
    :param k:近邻值
    :return:最多分类的类别
    '''
    m = dataset.shape[0]
    diffMat = np.tile(inX, (m, 1))-dataset
    sqdiffMat = diffMat**2
    sqdistances = sqdiffMat.sum(axis=1)
    distances = sqdistances**0.5
    sortedDistances = np.argsort(distances)
    classCount = {}
    # print(sortedDistances)
    #选k个距离中的最多分类
    for i in range(k):

        voteLabel = labels[sortedDistances[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def test():
    ratio = 0.1
    reMatrix, labelVec = file2matrix('data/datingTestSet2.txt')
    normDataset, ranges, minVals = autoNorm(reMatrix)
    m = normDataset.shape[0]
    testNum = int(m*ratio)
    print('testNum:', testNum)
    errorCount = 0.0
    for i in range(testNum):
        classifyResult = classify(normDataset[i, :], normDataset[testNum:m, :], labelVec[testNum:m], 3)
        print('predict result is %d, the real answer is %d' % (classifyResult, labelVec[i]))
        if classifyResult!=labelVec[i]:
            errorCount+=1
    print('the total error rate is %f' % (errorCount/m))
    print('the precious is %.2f' % ((m-errorCount)/m))
    print('done')

if __name__=='__main__':
    test()
