from collections import defaultdict
import os
import re
import json
import jieba
import codecs
import sys
import chardet
import matplotlib.pyplot as plt

def readlines(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        all_lines=[line.strip('\n') for line in f.readlines()]
    return all_lines


def sent2word(sentence):
    """
      segment a sentence to words
      Delete stopwords
    """
    segList = jieba.cut(sentence=sentence)
    segResult = []
    for w in segList:
        segResult.append(w)
    newSent = []
    stopwords = readlines('stop_words.txt')
    for word in segResult:
        if word in stopwords:
            continue
        else:
            newSent.append(word)
    return newSent


def classifyWords(wordDict):
    """
    ### 情感定位 ###
    :param wordDicts:
    :return:
    """
    # 情感词
    senList = readlines('./data/BosonNLP_sentiment_score.txt')
    senDict = defaultdict()
    for s in senList:
        s=s.replace('\n','')
        senDict[s.split(' ')[0]] = s.split(' ')[1]
    # 否定词
    notList = readlines('./data/notDict.txt')
    # 程度副词
    degreeList = readlines('./data/degreeDict.txt')
    degreeDict = defaultdict()

    for d in degreeList:
        degreeDict[d.split(',')[0]] = d.split(',')[1]
    senWord = defaultdict()
    notWord = defaultdict()
    degreeWord = defaultdict()

    for word in wordDict.keys():
        if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
            senWord[wordDict[word]] = senDict[word]
        elif word in notList and word not in degreeDict.keys():
            notWord[wordDict[word]] = -1
        elif word in degreeDict.keys():
            degreeWord[wordDict[word]] = degreeDict[word]
    return senWord, notWord, degreeWord


def scoreSent(senWord, notWord, degreeWord, segResult):
    """
    finalSentiScore = (-1) ^ (num of notWords) * degreeNum * sentiScore¶
    :param senWord:
    :param notWord:
    :param degreeWord:
    :param segResult:
    :return:
    """
    W = 1
    score = 0
    # 存储所有情感词的位置的列表
    senLoc = senWord.keys()
    notLoc = notWord.keys()
    degreeLoc = degreeWord.keys()
    wordLoc=segResult.values()#单词位置的列表
    senloc = -1
    # notLoc=-1
    # degreeLoc=-1

    for i in wordLoc:
        # 如果该词为情感词
        if i in senLoc:
            # loc为情感词位置列表的序号
            senloc += 1
            # 直接添加该情感词分数
            score += W * float(senWord[i])
            # print "score = %f" % score
            if senloc < len(senLoc) - 1:
                # 判断该情感词与下一情感词之间是否有否定词或程度副词
                # j为绝对位置
                for j in range(list(senLoc)[senloc], list(senLoc)[senloc + 1]):
                    # 如果有否定词
                    if j in notLoc:
                        W *= -1
                    # 如果有程度副词
                    elif j in degreeLoc:
                        W *= float(degreeWord[j])
        else:
            continue
        # if senloc < len(senLoc):
        #     i = list(senLoc)[senloc+1]
    return score

def fitler_stop(seg_list):
    final = defaultdict(dict)
    alllines = readlines('./data/stopwords.txt')
    stopwords = {}.fromkeys([line.rstrip() for line in alllines])
    i = 0
    for seg in seg_list:
        if seg not in stopwords:
            final[seg] = i
            i += 1
    return final

def VPrint(content):
    print(json.dumps(content,ensure_ascii=False,indent=4))

if __name__=="__main__":
    seg_list = jieba.cut("XXX是骗人的，我明明抽中了儿童滑板车，没过几分钟又说很遗憾我没有中奖，太可恶了", cut_all=False)
    # print "Default Mode:", "/ ".join(seg_list) # 精确模式
    seg_dict=fitler_stop(seg_list)
    print('句子')
    VPrint(seg_dict)
    [a, b, c] = classifyWords(seg_dict)
    print(" '情感词':%s,否定词:%s,程度词:%s"% (a,b,c))
    score = scoreSent(a, b, c, seg_dict)
    print('The sentence score:%s'% score)


