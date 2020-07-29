#! usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow_addons as tf_ad


class BiLSTM_CRF(tf.keras.Model):
    def __init__(self, vocab_size, label_size, hidden_dim, embedding_dim):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))
        self.dropout = tf.keras.layers.Dropout(0.5)

    # @tf.function
    def call(self, text, labels=None, training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        # -1 change 0
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        logits = self.dense(self.biLSTM(inputs))

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits,
                                                                                   label_sequences,
                                                                                   text_lens,
                                                                                   transition_params=self.transition_params)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens