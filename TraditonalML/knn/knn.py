import numpy as np
import operator
from os import listdir


def classify(input, dataSet, labels, k):
    """
    knn分类
    :param input:
    :param dataSet:
    :param label:
    :param k:
    :return:
    """
    dataSetSize = input.shape[0]
    diffMat = np.tile(input, (dataSetSize, 1)) - dataSet  # 测试数据与训练数据的距离
    sqDiffMat = diffMat ** 2
    sqdistance = sqDiffMat.sum(axis=1)
    distance = sqdistance ** 0.5
    sortedDistanceIndex = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistanceIndex[i]]
        classCount = classCount.get(voteLabel, 0) + 1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    """
      create data set
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

if __name__=="__main__":
    #create dataset and label
    dataSet,labels=createDataSet()
    #define test data
    testX=np.array([1,3,1.1])
    k=3
    outputlabel=classify(testX,dataSet,labels,k)

    print("The test data:%s classified to class:%s"%(testX,outputlabel))
