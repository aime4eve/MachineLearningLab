# encoding:utf-8
"""
RNN-LSTM循环神经网络
"""

import tensorflow as tf
def network_model(inputs,num_pitch,weights_file=None):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.LSTM(
        512,
        inputs_shape=(inputs.shape[1],inputs.shape[2]),
        return_sequences=True
    ))

    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512,return_sequence=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512))
    model.add(tf.keras.layers.Dense(256))  # 256 个神经元的全连接层
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_pitch))  # 输出的数目等于所有不重复的音调的数目
    model.add(tf.keras.layers.Activation('softmax'))  # Softmax 激活函数算概率

    # 交叉熵计算误差，使用对 循环神经网络来说比较优秀的 RMSProp 优化器
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    if weights_file is not None:
        model.load_weights(weights_file)
    return model
