import random
import numpy as np
from utils import *

class kMeans(object):

    def __init__(self,dataSet,k):
        self.dataSet=dataSet
        self.k=k

    def _randCenters(self,dataSet,k):
        """
         # 随机生成聚类中心
        :return:
        """
        n = np.shape(dataSet)[1]
        clustercenter=np.mat(np.zeros((k,n)))
        for col in range(n):
            mincol=min(self.dataSet[:,col])[0,0]
            maxcol=max(self.dataSet[:,col])[0,0]
            randomdata=(maxcol-mincol)*np.random.random(size=(k,1))+mincol
            clustercenter[:,col]=randomdata
        return clustercenter

    def _distEclud(self,vecA,vecB):
        """
        欧式距离
        :param vecA:
        :param vecB:
        :return:
        """
        return np.linalg.norm(vecA-vecB)

    def train(self):
        m = np.shape(self.dataSet)[0]  # 返回矩阵的行数
        # 列1：数据集对应的聚类中心,列2:数据集行向量到聚类中心的距离
        ClustDist = np.mat(np.zeros((m, 2)))
        # 随机生成一个数据集的聚类中心:本例为4*2的矩阵
        # 确保该聚类中心位于min(dataSet[:,j]),max(dataSet[:,j])之间
        clustercents = self._randCenters(self.dataSet,self.k)  # 随机生成聚类中心

        flag = True
        counter = []

        # stop:dataSet的所有向量都能找到某个聚类中心,到此中心的距离均小于其他k-1个中心的距离
        while flag:
            flag = False  # 预置标志位为False
            # 将此结果赋值ClustDist=[minIndex,minDist]
            for i in range(m):
                # 遍历k个聚类中心,获取最短距离
                distlist = [self._distEclud(clustercents[j, :], self.dataSet[i, :]) for j in range(self.k)]
                minDist = min(distlist)
                minIndex = distlist.index(minDist)

                if ClustDist[i, 0] != minIndex:  # 找到了一个新聚类中心
                    flag = True  # 重置标志位为True，继续迭代

                # 将minIndex和minDist**2赋予ClustDist第i行
                # 含义是数据集i行对应的聚类中心为minIndex,最短距离为minDist
                ClustDist[i, :] = minIndex, minDist

            #   更新clustercents值: 循环变量为cent(0~k-1)
            # 1.用聚类中心cent切分为ClustDist，返回dataSet的行索引
            # 并以此从dataSet中提取对应的行向量构成新的ptsInClust
            # 计算分隔后ptsInClust各列的均值，以此更新聚类中心clustercents的各项值
            for cent in range(self.k):
                # 从ClustDist的第一列中筛选出等于cent值的行下标
                dInx = np.nonzero(ClustDist[:, 0].A == cent)[0]
                ptsInClust = self.dataSet[dInx]
                # 计算ptsInClust各列的均值: mean(ptsInClust, axis=0):axis=0 按列计算
                clustercents[cent, :] = np.mean(ptsInClust, axis=0)
        return clustercents, ClustDist
