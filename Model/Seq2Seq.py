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
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Dense, LSTM, Concatenate, Reshape, Embedding, Input, Bidirectional


class BahdanauAttention(Layer):
    def __init__(self, units, name='attention_layer', **kwargs):
        super(BahdanauAttention, self).__init__(name=name, **kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)


    def get_config(self):
        # config = {"units": self.units}
        base_config = super(BahdanauAttention, self).get_config()

        return dict(list(base_config.items()))

    def call(self, inputs):
        query, values = inputs[0], inputs[1]
        # hidden shape == (batch_size, decoder_units)
        # hidden_with_time_axis shape == (batch_size, 1, decoder_units)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # score shape == (batch_size, encoder_seq_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, encoder_seq_len, decoder_units)
        score = self.V(tf.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, encoder_seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, encoder_units)
        # values shape (batch_size, encoder_seq_len, encoder_units)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Encoder(Model):
    def __init__(self, vocab_size, embed_dim, unit_num):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.unit_num = unit_num
        self.embedding = Embedding(vocab_size, embed_dim, embeddings_initializer="uniform", name="encoder_embed")
        forward_lstm = LSTM(self.unit_num, return_sequences=True, go_backwards=False, dropout=0.4,
                                            recurrent_initializer="glorot_uniform", name="forward_lstm")
        backward_lstm = LSTM(self.unit_num, return_sequences=True, go_backwards=True, dropout=0.4,
                                             return_state=True, name="backward_lstm")
        self.bilstm = Bidirectional(merge_mode="concat", layer=forward_lstm, backward_layer=backward_lstm,
                                    name="ecnoder_bilstm")


    def call(self, enc_input, training):
        enc_embed = self.embedding(enc_input)
        enc_output, enc_hid = self.bilstm(enc_embed)
        return enc_output, enc_hid


class Decoder(Model):
    def __init__(self, vocab_size, embed_dim, unit_num):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.unit_num = unit_num
        self.embed = Embedding(vocab_size, embed_dim, embeddings_initializer="uniform", name="decoder_embed")
        self.lstm = LSTM(self.unit_num, return_sequences=True, return_state=True, dropout=0.4,
                         recurrent_initializer="glorot_uniform", name="decoder_lstm")
        self.attention = BahdanauAttention(self.unit_num)

    def call(self, dec_input, enc_hid, enc_output):
        context_vector, attention_weight = self.attention(enc_hid, enc_output)
        dec_embed = self.embed(dec_input)
        dec_output = Concatenate(axis=-1)([tf.expand_dims(context_vector, 1), dec_embed])
        dec_output, dec_state = self.lstm(dec_output)
        dec_output = Reshape(target_shape=(-1, self.unit_num))
        dec_output = Dense(self.vocab_size)
        return dec_output, dec_state, attention_weight


def loss_function(targets, preds, padding_value):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
    # Ignore the padding value.
    mask = tf.math.logical_not(tf.math.equal(targets, padding_value))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = cross_entropy(targets, preds, sample_weight=mask)

    return loss


# 一个比较好的解决方法就是将“O”，也就是Padding前置，即16
@tf.function
def train_step(input, target, optimizer, encoder, decoder, padding_value, target_dic):
    loss = 0
    with tf.GradientTape() as tape:
        encoder_out, encoder_hid = encoder(input)
        decoder_hid = encoder_hid
        decoder_in = tf.expand_dims([16] * BATCH_SIZE, 1)
        # feeding the target as the next input
        for t in range(1, target.shape[1]):
            # passing encoder_output to the decoder
            predictions, decoding_hidden = decoder([decoder_in, decoder_hid, encoder_out])
            loss += loss_function(target[:, t], predictions, padding_value)
            decoder_in = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[0]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def infer(base_vec, encoder, decoder):
    pred_array = []
    for k in range(len(base_vec)):
        word = np.array(base_vec[k])
        word = np.reshape(word, (1, 16))
        word = tf.convert_to_tensor(word)
        hidden = tf.zeros((1,))
        encoder_out, encoder_hid = encoder(word, hidden)
        decoder_hid = encoder_hid
        # decoder_input = tf.expand_dims([0], 0)
        decoder_input = tf.expand_dims([16], 0) # 因为padding是O，前置的padding，所以是16

        result = []
        for i in range(10):
            pred, decoder_hid = decoder([decoder_input, decoder_hid, encoder_out])
            pred = tf.squeeze(pred)
            pred_id = tf.argmax(pred).numpy()
            if pred_id == 1:
                break
            result.append(pred_id)
            decoder_input = tf.expand_dims([pred_id], 0)
        result = np.array(result)
        pred_array.append(result)

    return pred_array


if __name__ == "__main__":
    pass