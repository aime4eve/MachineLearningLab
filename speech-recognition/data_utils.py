import numpy as np
import scipy.io.wavfile as wav
import os
from scipy.fftpack import fft
import matplotlib.pyplot as plt

#获取信号的时频图
def compute_freqbank(filepath):
    x=np.linspace(0,400-1,400,dtype=np.int64)
    w=0.54-0.46*np.cos(2*np.pi*(x)/(400-1)) #加汉名窗
    fs,wavsignal=wav.read(filepath)
    # wav波形 加时间窗以及时移10ms
    time_window=25 # 单位ms
    window_length = fs / 1000 * time_window  # 计算窗长度的公式，目前全部为400固定值
    wav_arr=np.array(wavsignal)
    wav_length=len(wav_arr)
    range0_end=int(len(wavsignal)/fs*1000-time_window) // 10 #计算循环终止的位置，也就是最终生成的窗数
    data_input=np.zeros((range0_end,200),dtype=np.float) # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0,range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
	data_input = np.log(data_input + 1)
	#data_input = data_input[::]
	return data_input

def label_padding(label_data_lst):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens


def get_batch(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(10000//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_freqbank(wav_lst[index])
            fbank = fbank[:fbank.shape[0] // 8 * 8, :]
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(fbank)
            label_data_lst.append(label)
        yield wav_data_lst, label_data_lst


def wav_padding(wav_data_lst):
    """
    构成一个tensorflow块，要求每个样本数据形式是一样
    :param wav_data_lst:
    :return:
    """
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng//8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens

def data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(len(wav_lst)//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_freqbank(wav_lst[index])
            pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))
            pad_fbank[:fbank.shape[0], :] = fbank
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(pad_fbank)
            label_data_lst.append(label)
        pad_wav_data, input_length = wav_padding(wav_data_lst)
        pad_label_data, label_length = label_padding(label_data_lst)
        inputs = {'the_inputs': pad_wav_data,
                  'the_labels': pad_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                 }
        outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)}
        yield inputs, outputs

def source_get(source_file):
    train_file=source_file+'/data'
    label_lst=[]
    wav_lst=[]
    for root,dirs,files in os.walk(train_file):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                wav_file=os.sep.join([root,file])
                label_file=wav_file+'.trn'
                wav_lst.append(label_file)
    return label_lst,wav_lst

def read_label(label_file):
    with open(label_file,'r',encoding='utf-8') as f:
        data=f.readlines()
        return data[1]
def gen_label_data(label_lst):
    label_data = []
    for label_file in label_lst:
        pny = read_label(label_file)
        label_data.append(pny.strip('\n'))
    return label_data

def mk_vocab(label_data):
    vocab = []
    for line in label_data:
        line = line.split(' ')
        for pn in line:
            if pn not in vocab:
                vocab.append(pn)
        vocab.append('_')
    return vocab

def word2id(line, vocab):
    """
    label映射到对应的id
    :param line:
    :param vocab:
    :return:
    """
    return [vocab.index(pny) for pny in line.split(' ')]


def get_build_data(file_path):
    source_file = file_path
    label_lst, wav_lst = source_get(source_file)

    label_data = gen_label_data(label_lst[:100])

    vocab = mk_vocab(label_data)
    vocab_size = len(vocab)
    print(vocab_size)
    return wav_lst, label_data, vocab, vocab_size


if __name__=="__main__":
    filepath = 'test.wav'

    a = compute_freqbank(filepath)
    plt.imshow(a.T, origin='lower')
    plt.show()

    source_file=''
    label_lst, wav_lst = source_get(source_file)

    print(label_lst[:10])
    print(wav_lst[:10])