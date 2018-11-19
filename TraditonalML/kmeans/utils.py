import matplotlib as plt
import os
import numpy as np


def drawScatter(plt,dataset,size=20,color='blue',mrkr='0'):
    """
    绘制散点图
    :param plt:
    :param dataset:
    :param size:
    :param color:
    :param mrkr:
    :return:
    """
    plt.scatter(dataset[0].tolist(), dataset[1].tolist(), s=size, c=color, marker=mrkr)


def color_cluster(dataindx, dataSet, plt):
    """
    以不同颜色绘制数据集里的点
    :param dataindx:
    :param dataSet:
    :param plt:
    :return:
    """
    datalen = len(dataindx)
    for indx in range(datalen):
        if int(dataindx[indx]) == 0:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], c='blue', marker='o')
        elif int(dataindx[indx]) == 1:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], c='green', marker='o')
        elif int(dataindx[indx]) == 2:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], c='red', marker='o')
        elif int(dataindx[indx]) == 3:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], c='cyan', marker='o')

def strdata2num(str):
    if len(str)>2:
        return float(str)
    else:
        return int(str)
def file2matrix(path,delimiter):
    """
    # 数据文件转矩阵
    :param path:
    :param delimiter:
    :return:
    """
    with open(path,'r') as f:
        rowlist=f.readlines()
        recordlist=[list(map(strdata2num,row.replace('\n','').split(delimiter))) \
                    for row in rowlist if row.strip()]
        return np.mat(recordlist)
