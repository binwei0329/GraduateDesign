#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Graduate Project: Named Entity Recognition Task in Chinese Media
Author: Bin Wei
Description: This is part of my graduate project, which tries to explore some
            models targeting NER tasks in Chinese social media.
"""

import tensorflow as tf
from tensorflow_addons.layers import CRF
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Conv1D

class IDCNNCRF(object):
    def __init__(self, vocab_size, tag_size, max_len, embedding_dim, drop_rate):
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.drop_rate = drop_rate


    def creat_model(self):
        """
        本网络的机构采用的是，
           Embedding
           直接进行2个常规一维卷积操作
           接上一个空洞卷积操作
           连接全连接层
           最后连接CRF层
        kernel_size 采用2、3、4
        cnn  特征层数: 64、128、128
        """

        inputs = Input(shape=(self.max_len,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',dilation_rate=2)(x)

        x = Dropout(self.drop_rate)(x)
        x = Dense(self.tag_size)(x)
        self.crf = CRF(self.tag_size, sparse_target=False)
        x = self.crf(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()
        self.compile()
        return self.model


    def compile(self):
        self.model.compile('adam',
                           loss=self.crf.loss_function,
                           metrics=[self.crf.accuracy])