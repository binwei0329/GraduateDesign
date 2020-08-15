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


# def loss(labels, logits):
#   return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
#
# example_batch_loss  = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())


def loss_function(gold_seq, pred_seq):
# def loss_function(targets, preds, padding_value):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
    # Ignore the padding value.
    # mask = tf.math.logical_not(tf.math.equal(targets, padding_value))
    # mask = tf.cast(mask, dtype=tf.int64)
    loss = cross_entropy(gold_seq, pred_seq)

    return loss


# 一个比较好的解决方法就是将“O”，也就是Padding前置，即16
@tf.function
def train_step(input_seq, target_seq, optimizer, encoder, decoder):
# def train_step(input_seq, output_seq, optimizer, encoder, decoder, padding_value, target_dic, batch_size):
    loss = 0
    batch_size = int(target_seq[0])
    with tf.GradientTape() as tape:
        enc_out, enc_hid = encoder(input_seq)
        dec_hid = enc_hid
        dec_in = tf.expand_dims([16] * batch_size, 1)

        # feeding the target as the next input
        for t in range(1, target_seq.shape[1]):
            # passing encoder_output to the decoder
            pred_seq, dec_hid = decoder([dec_in, dec_hid, enc_out])

            # loss += loss_function(target[:, t], predictions, padding_value)
            loss += loss_function(target_seq[:, t], pred_seq)

            dec_in = tf.expand_dims(target_seq[:, t], 1)

    # batch_loss = (loss / int(target_seq.shape[0]))
    batch_loss = (loss / batch_size)
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def predict(input_seq, encoder, decoder):
    pred_sequence = []
    for k in range(len(input_seq)):
        sentence = np.array(input_seq[k])
        sentence = np.reshape(sentence, (1, 256))
        # word = np.reshape(word, (1, 16))

        sentence = tf.convert_to_tensor(sentence)
        hidden = tf.zeros((1,))
        enc_out, enc_hid = encoder(sentence, hidden)
        dec_hid = enc_hid
        # decoder_input = tf.expand_dims([0], 0)
        dec_in = tf.expand_dims([16], 0) # 因为padding是O，前置的padding，所以是16

        result = []
        for i in range(10):
            pred_seq, dec_hid = decoder([dec_in, dec_hid, enc_out])
            pred_seq = tf.squeeze(pred_seq)
            pred_id = tf.argmax(pred_seq).numpy()
            if pred_id == 1:
                break
            result.append(pred_id)
            dec_in = tf.expand_dims([pred_id], 0)
        result = np.array(result)
        pred_sequence.append(result)

    return pred_sequence


if __name__ == "__main__":
    pass