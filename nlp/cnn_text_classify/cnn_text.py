#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

class TextCNN(object):

    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,\
                 filter_sizes,num_filters,l2_reg_lambda=0.0):
        #input ,output, and dropout
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        #12 regulariztion loss
        l2_loss=tf.constant(0.0)
        #Embeding layer
        with tf.device('/cpu:0'),tf.name_scope("embeding"):
            self.W=tf.Variable(
                tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),
                name="W"
            )
            self.embedded_char = tf.nn.embedding_lookup(self.W,self.input_x)
            self.embedded_char_expanded=tf.expand_dims(self.embedded_chars,-1)

            #Create a convolution + maxpool layer for each filter size

            pooled_outputs=[]

            for i,filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    #Conv layer
                    filter_shape=[filter_size,embedding_size,1,num_filters]
                    W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                    b= tf.Variable(tf.constant(0.1,shape=[num_filters],name="b"))
                    conv=tf.nn.conv2d(self.embedded_char_expanded,
                                      W,
                                      strides=[1,1,1,1],
                                      padding="VALD",
                                      name="conv")
                    #Apply nonlinearity
                    h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")

                    #Mappooling over the outputs

                    pooled=tf.nn.max_pool(
                        h,
                        ksize=[1,sequence_length-filter_size+1,1,1],
                        strides=[1,1,1,1],
                        name="pool"
                    )
                    pooled_outputs.append(pooled)

            #Combine all the pooled features
            num_filters_total=num_filters*len(filter_sizes)
            self.h_pool=tf.concat(pooled_outputs,3)
            #flat matix
            self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])

            #add dropout
            with tf.name_scope("dropout"):
                self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

            #final scores and predictions

            with tf.name_scope("output"):
                W=tf.get_variable(
                    "W",
                    shape=[num_filters_total,num_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                b=tf.Variable(tf.constant(0.1,shape=[num_classes],name="b"))
                l2_loss+=tf.nn.l2_loss(W)
                l2_loss+=tf.nn.l2_loss(b)

                self.scoers=tf.nn.xw_plus_b(self.h_drop,W,b,name="scores")
                self.predictions=tf.arg_max(self.scoers,1,name="predictions")
            #Calculate mean cross-entry loss
            with tf.name_scope("loss"):
                losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scoers,labels=self.input_y)
                self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss
                # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")




