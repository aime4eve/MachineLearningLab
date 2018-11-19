from sklearn.cross_validation import  train_test_split
from sklearn import cross_validation,metrics
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC

def load_file_and_processing():
    neg = pd.read_excel('./data/neg.xls', header=None, index=None)
    pos = pd.read_excel('./data/pos.xls', header=None, index=None)

    cw = lambda x : list(jieba.cut(x))

    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    #构建label数据
    y=np.concatenate((np.ones(len(pos)),np.zeros(len(neg))),axis=0)

    x_train,x_test,y_train,y_test=train_test_split(np.concatenate((pos['words'],neg['words'])),y,test_size=0.3)

    np.save('./svm_data/y_train.npy',y_train)
    np.save('./svm_data/y_test.npy',y_test)

    return x_train,x_test

def build_word2vec(text,size,imdb_w2v):
    """
      每个句子的所有词向量取均值，来生成一个句子的vector
    :param text:
    :param size:
    :param imdb_w2v:
    :return:
    """
    vec = np.zeros(size).reshape((1,size))

    count=0
    for word in text:
        try:
            vec+=imdb_w2v[word].reshape((1,size))
            count+=1.
        except KeyError as e:
            continue
    if count!=0:
        vec/=count
    return vec

def get_train_test_vecs(x_train,x_test):
    n_dim=300
    # 初始化模型和词表
    imdb_w2v=Word2Vec(x_train,size=n_dim,min_count=10)
    train_vecs=np.concatenate([build_word2vec(z,n_dim,imdb_w2v) for z in x_train])
    np.save('./svm_data/train_vecs.npy', train_vecs)

    # imdb_w2v.build_vocab(x_train)
    # imdb_w2v.train(x_train,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)
    print(train_vecs.shape)
    #imdb_w2v.train(x_test)
    imdb_w2v.train(x_test,
                   total_examples=imdb_w2v.corpus_count,
                   epochs=imdb_w2v.iter)
    imdb_w2v.save('./svm_data/w2v_model/w2v_model.pkl')

    test_vec=np.concatenate([build_word2vec(z,n_dim,imdb_w2v) for z in x_test])
    np.save('./svm_data/test_vecs.npy',test_vec)

def get_data():
    train_vecs=np.load('./svm_data/train_vecs.npy')
    y_train=np.load('./svm_data/y_train.npy')
    test_vecs=np.load('./svm_data/test_vecs.npy')
    y_test=np.load('./svm_data/y_test.npy')
    return train_vecs,y_train,test_vecs,y_test

def svm_model_train(train_vecs,y_train,test_vecs,y_test):
    model=SVC(kernel='rbf',verbose=True)
    model.fit(train_vecs,y_train)
    joblib.dump( model, './svm_data/svm_model/model.pkl')
    print("the model accuracy:%s"% model.score(test_vecs, y_test))
    predict_prob_y=model.predict(test_vecs)
    test_auc = metrics.roc_auc_score(y_test,predict_prob_y)
    print("the model auc:%s" % test_auc)


def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('./svm_data/w2v_model/w2v_model.pkl')
    #imdb_w2v.train(words)
    train_vecs = build_word2vec(words, n_dim,imdb_w2v)
    #print train_vecs.shape
    return train_vecs

def svm_predict(string):
    """
    单个句子进行情感判断
    :param string:
    :return:
    """
    words = jieba.lcut(string)

    words_vecs=get_predict_vecs(words)

    model=joblib.load('./svm_data/svm_model/model.pkl')
    result=model.predict(words_vecs)

    if int(result[0])==1:
        print('the sentent %s is positive' % string)
    else:
        print('the sentent %s is negative' % string)

#
x_train,x_test = load_file_and_processing()
get_train_test_vecs(x_train,x_test)
train_vecs,y_train,test_vecs,y_test = get_data()
svm_model_train(train_vecs,y_train,test_vecs,y_test)

##对输入句子情感进行判断
#string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
#string='这手机真棒，从1米高的地方摔下去就坏了'
#svm_predict(string)