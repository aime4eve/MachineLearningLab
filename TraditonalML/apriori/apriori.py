
"""
   @author:nicajonh
"""

import numpy as np

def loadDataSet():

    return [[1,3,4],[1,3,4,7], [2,3,6,5], [1,2,7,3,5], [2,5,6],[1,2,3,6]]

def createC1(dataSet):
    """
      generate C1

    :param dataSet:
    :return:
    """
    C1=[]
    for transaciton in dataSet:
        for item in transaciton:
            if not [item] in C1:
                C1.append(item)
    C1=list(set(C1))
    C1.sort()
    return list(map(lambda x:frozenset([x]), C1))

def scanD(DataSet,Ck,miniSupport):
    """
     计算Ck中每项的支持度并过滤
    :param D:
    :param Ck:
    :param minSupport:
    :return:
    """
    ssCnt={}
    for datas in DataSet:
        for c in Ck:
            if c.issubset(datas):
                if c not in ssCnt:
                    ssCnt[c]=1
                else:
                    ssCnt[c]+=1
    #filter support
    retList=[]
    supportData={}#支持度
    for key in ssCnt:
        support=ssCnt[key]/float(len(DataSet))
        if support>miniSupport:
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            lk1=Lk[i]
            lk2=Lk[j]
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2: # 如果它们前k-2项相同
                retList.append(Lk[i] | Lk[j]) # 合并
    return retList
def apriori(dataSet,miniSupport=0.5):
    C1=createC1(dataSet)
    D =list(map(set,dataSet))
    L1,supportData=scanD(D,C1,miniSupport)
    L=[L1]
    k=2
    while len(L[k-2])>0:
        Ck=aprioriGen(L[k-2],k)
        Lk,supk=scanD(D,Ck,miniSupport)
        supportData.update(supk)
        L.append(Lk)
        k+=1
    return L,supportData

# def calcConf(freqSet, H, supportData, br1, minConf=0.7):
#     prunedH = []
#     for conseq in H:
#         count1=freqSet-conseq
#         count2=supportData[count1]
#         conf = supportData[freqSet] / supportData[freqSet - conseq]
#         if conf >= minConf:
#             # 当label被包含在H中
#             if "1" in conseq or "0" in conseq:
#                 # print "{0} --> {1} conf:{2}".format(freqSet - conseq, conseq, conf)
#                 br1.append((freqSet - conseq, conseq, conf))
#                 prunedH.append(conseq)
#     return prunedH

def calcConf(freqSet,H,supportData,br1,miniConf=0.7):
    """
    计算可信度
    :return:
    """
    prunedH=[]
    for conseq in H:
        conf=supportData[freqSet]/supportData[freqSet-conseq]
        if conf>=miniConf:
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > m+1:
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if len(Hmp1)>1:
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

def generateRules(L, supportData, minConf=0.7):
    """
    生成规则表
    :param L:
    :param supportData:
    :param minConf:
    :return:
    """
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i>1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


if __name__=="__main__":
    L, suppData = apriori(loadDataSet(), 0.25)
    # 生成规则，每个规则的置信度至少是0.6
    bigRuleList = generateRules(L, suppData, 0.6)
    dataSet=loadDataSet()
    C=createC1(dataSet)
    retlist,supportData=scanD(dataSet,C,miniSupport=0.2)
    print(retlist,supportData)

