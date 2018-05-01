import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


#超参数
PAD = 0 # 补全字符
EOS = 1 # 解码器端的结束标识各个符
vocab_size = 10 #字符串大小
input_embedding_size = 20#词嵌入大小
encoder_hidden_units = 20
decoder_hidden_units = 20
batch_size = 100
def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
            for _ in range(batch_size)]


batches = random_sequences(length_from=3, length_to=10,
                           vocab_lower=2, vocab_upper=10,
                           batch_size=batch_size)



def make_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths


train_grahp=tf.Graph()

with train_grahp.as_default():
    encoder_inputs=tf.placeholder(shape=(None,None),dtype=tf.int32,name='encoder_inputs')
    decoder_inputs=tf.placeholder(shape=(None,None),dtype=tf.int32,name='decoder_input')
    decoder_targets=tf.placeholder(shape=(None,None),dtype=tf.int32,name='decoder_target')
    embeddings= tf.Variable(tf.random_uniform([vocab_size,input_embedding_size],-1.0,1.0),\
                            dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,decoder_inputs)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    encoder_outputs,encoder_final_state=tf.nn.dynamic_rnn(encoder_cell,encoder_inputs_embedded,\
                                                          dtype=tf.float32,time_major=True)
    decoder_cell=tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    decoder_outputs,decoder_final_state=tf.nn.dynamic_rnn(decoder_cell,decoder_inputs_embedded,\
                                                          initial_state=encoder_final_state,dtype=tf.float32,time_major=True,\
                                                          scope="plain_decoder")

    decoder_logits=tf.contrib.layers.linear(decoder_outputs,vocab_size)
    decoder_prediction = tf.argmax(decoder_logits,axis=2)
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets,\
                            depth=vocab_size,dtype=tf.float32),logits=decoder_logits)

    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)

loss_track = []
epochs = 3001

#train code