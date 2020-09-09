#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Graduate Project: Named Entity Recognition Task in Chinese Media
Author: Bin Wei
Description: This is part of my graduate project, which tries to explore some
            models targeting NER tasks in Chinese social media.
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tf_ad
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D

class IDCNN_CRF(tf.keras.Model):
    def __init__(self, vocab_size, tag_size, embed_dim):
        super(IDCNN_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embed_dim = embed_dim

        self.embedding = Embedding(vocab_size, embed_dim)
        self.conv1 = Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', dilation_rate=1)
        self.conv2 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same', dilation_rate=1)
        self.conv3 = Conv1D(filters=512, kernel_size=4, activation='relu', padding='same',dilation_rate=2)

        self.dense = Dense(self.tag_size)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(tag_size, tag_size)))
        self.dropout = Dropout(0.5)


    def call(self, text, labels=None, training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        inputs = self.embedding(text)
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.conv3(inputs)
        inputs = self.dropout(inputs, training)
        logits = self.dense(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits,
                                                    label_sequences, text_lens, transition_params=self.transition_params)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens

    # def creat_model(self):
    #     """
    #     本网络的机构采用的是，
    #        Embedding
    #        直接进行2个常规一维卷积操作
    #        接上一个空洞卷积操作
    #        连接全连接层
    #        最后连接CRF层
    #     kernel_size 采用2、3、4
    #     cnn  特征层数: 64、128、128
    #     """
    #
    #     inputs = Input(shape=(self.max_len,))
    #     x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
    #     x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(x)
    #     x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(x)
    #     x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',dilation_rate=2)(x)
    #
    #     x = Dropout(self.drop_rate)(x)
    #     x = Dense(self.tag_size)(x)
    #     self.crf = CRF(self.tag_size, sparse_target=False)
    #     x = self.crf(x)
    #     self.model = Model(inputs=inputs, outputs=x)
    #     self.model.summary()
    #     self.compile()
    #     return self.model
    #
    #
    # def compile(self):
    #     self.model.compile('adam',
    #                        loss=self.crf.loss_function,
    #                        metrics=[self.crf.accuracy])


def train_one_step(model, data, label, opt):
    with tf.GradientTape() as tape:
        logits, text_lens, log_likelihood = model(data, label, training=True)
        loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, logits, text_lens

def predict(model, labels, data):
    """
    This method uses the trained model to generate predictions.
    :param model: trained model
    :param labels: gold labels
    :param data: data
    :return: the predictions
    """
    predictions = []
    logits, text_lens, log_likelihood = model(data, labels)
    for logit, text_len, label in zip(logits, text_lens, labels):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        viterbi_path = np.array(viterbi_path)
        predictions.append(viterbi_path)

    return predictions