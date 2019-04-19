import keras
from keras.layers import Input,Conv2D,BatchNormalization,MaxPooling2D
from keras.layers import Reshape,Dense,Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.models import load_model
from .model_base import *
from .data_utils import *



class Amodel(object):
    """
    搭建cnn+dnn+ctc的声学模型
    """

    def __init__(self,vocab_size):
        super(Amodel, self).__init__()
        self.vocab_size=vocab_size
        self._model_init()
        self._ctc_init()
        self.opt_init()

    def _model_init(self):
        self.inputs=Input(name='the_inputs',shape=(None,200,1))
        self.h1=cnn_cell(32,self.inputs)
        self.h2=cnn_cell(64,self.h1)
        self.h3=cnn_cell(128,self.h2)
        self.h4=cnn_cell(128,self.h3,pool=False)
        # 200 / 8 * 128 = 3200
        self.h6=Reshape((-1,3200))(self.h4)
        self.h7 =  dense(256)(self.h6)
        self.outputs=dense(self.vocab_size,activation='softmax')(self.h7)
        self.model=Model(inputs=self.inputs,outputs=self.outputs)

    def _ctc_init(self):
        self.labels=Input(name='the_labels',shape=[None],dtype='float32')
        self.input_length=Input(name='input_length',shape=[1],dtype='int64')
        self.label_length=Input(name='label_length',shape=[1],dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc') \
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
                                       self.input_length, self.label_length], outputs=self.loss_out)
    def opt_init(self):
        opt=Adam(lr=0.0008,beta_1=0.9,beta_2=0.999,decay=0.01,epsilon=10e-8)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)

def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1];
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text

def train_model():
    total_nums = 100
    batch_size = 20
    batch_num = total_nums // batch_size
    epochs = 50
    wav_lst, label_data, vocab,vocab_size = get_build_data('data_thchs30')
    shuffle_list = [i for i in range(100)]

    am = Amodel(vocab_size)

    for k in range(epochs):
        print('this is the', k + 1, 'th epochs trainning !!!')
        # shuffle(shuffle_list)
        batch = data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab)
        am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)
    am.ctc_model.save('Amodel.h5')


def test_model():
    # 测试模型 predict(x, batch_size=None, verbose=0, steps=None)
    shuffle_list = [i for i in range(100)]

    wav_lst,label_data,vocab,vocab_size=get_build_data('data_thchs30')

    batch = data_generator(1, shuffle_list, wav_lst, label_data, vocab)
    am=load_model('Amodel.h5')
    for i in range(10):
        # 载入训练好的模型，并进行识别
        inputs, outputs = next(batch)
        x = inputs['the_inputs']
        y = inputs['the_labels'][0]
        result = am.model.predict(x, steps=1)
        # 将数字结果转化为文本结果
        result, text = decode_ctc(result, vocab)
        print('数字结果： ', result)
        print('文本结果：', text)
        print('原文结果：', [vocab[int(i)] for i in y])