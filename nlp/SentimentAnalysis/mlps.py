from __future__ import absolute_import
from __future__ import division,print_function
import pandas as pd
import numpy as np
import jieba
from keras.preprocessing import sequence
from keras.optimizers import SGD,RMSprop,Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dropout,Dense,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU




def process_data():

    neg = pd.read_excel('./data/neg.xls',header=None,index=None)
    pos = pd.read_excel('./data/pos.xls',header=None,index=None)#读取训练语料完毕
    pos['mark']=1
    neg['mark']=0
    pn=pd.concat([pos,neg],ignore_index=True) #合并预
    cw=lambda x:list(jieba.cut(x)) #定义分词函数
    pn['words'] = pn[0].apply(cw)
    comment = pd.read_excel('./data/sum.xls') #读入评论内容
    #comment = pd.read_csv('a.csv', encoding='utf-8')
    comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
    comment['words'] = comment['rateContent'].apply(cw) #评论分词
    d2v_train=pd.concat([pn['words'],comment['words']],ignore_index=True)
    corpus = [] #将所有词语整合在一起
    for i in d2v_train:
        corpus.extend(i)
    return corpus,pn

def buil_model(x_data,y_data,batchsize=16,epoch=10):
    model=Sequential()
    model.add(Embedding(len(dict)+1,256))
    model.add(LSTM(256,128))
    model.add(Dropout(0.5))
    model.add(Dense(128,1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', class_mode="binary")
    model.fit(x_data,y_data,batch_size=batchsize,nb_epoch=epoch)
    return model


w,pn=process_data()
dict=pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数

x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))
#del w,d2v_train

dict['id'] = list(range(1,len(dict)+1))

get_sent= lambda x:list(dict['id'][x])

pn['sent']=pn['words'].apply(get_sent)

maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

model=buil_model(x,y)
classes=model.predict_classes(xt)
acc=np_utils.accuracy(classes,yt)
print('Test Accuray is:%s' % acc)