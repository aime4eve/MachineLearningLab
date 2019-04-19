from keras.layers import Conv2D,BatchNormalization,MaxPooling2D
from keras.layers import Dense
from keras import backend as K

def conv2d(size):
    return Conv2D(size,(3,3),use_bias=True,activation='rule',padding='same',kernel_initializer='he_normal')

def norm(x):
    return BatchNormalization(axis=-1)(x)

def maxpool(x):
    return MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')(x)

def dense(units,activation='relu'):
    return Dense(units,activation=activation,use_bias=True,kernel_initializer='he_normal')


def cnn_cell(size,x,pool=True):
    """
    cnn + cnn + maxpool结构
    :param size:
    :param x:
    :param pool:
    :return:
    """
    x=norm(conv2d(size)(x))
    x=norm(conv2d(size)(x))
    if pool:
        x=maxpool(x)
    return x

def ctc_lambda(args):
    """
    添加CTC损失函数，由backend引入
    labels 标签：[batch_size, l]
    y_pred cnn网络的输出：[batch_size, t, vocab_size]
    input_length 网络输出的长度：[batch_size]
    label_length 标签的长度：[batch_size]
    :param args:
    :return:
    """
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)