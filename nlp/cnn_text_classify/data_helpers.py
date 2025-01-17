import re
import numpy as np
import itertools
from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_labels(positive_data_file,negative_data_file):
    """
       Loads MR polarity data from files, splits the data into words and generates labels.
       Returns split sentences and labels.
       """

    # loader data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels=[[0,1] for _ in positive_examples]
    negative_labels=[[1,0] for _ in positive_examples]

    y= np.concatenate([positive_labels,negative_labels],0)

    return [x_text,y]

def batch_iter(data,batch_size,num_epochs,shuffle=True):

    """
     Generates a batch iterator for a dataset.
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """

    data = np.array(data)
    data_size=len(data)
    num_batch_per_epoch= int((len(data)-1)/batch_size)+1

    for epoch in range(num_epochs):
        #shuffle the data each epoch
        if shuffle:
            shuffle_indices= np.random.permutation(np.arange(data_size))
            shuffle_data=data[shuffle_indices]
        else:
            shuffle_data=data

        for batch_num in range(num_batch_per_epoch):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size,data_size)
            yield shuffle_data[start_index:end_index]
