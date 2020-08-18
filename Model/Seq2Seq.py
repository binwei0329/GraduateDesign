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


def build_encoder(encoder_input, vocab_size, embed_dim, unit_size):
    # shape: (batch_size, inputs.shape[1], embed_dim)
    encoder_embed = Embedding(vocab_size, embed_dim)(encoder_input)
    # shape: (batch_size, inputs.shape[1], encoder_units), (batch_size, encoder_units)
    encoder_output, encoder_hid, encoder_cell = LSTM(unit_size, return_sequences=True, return_state=True, dropout=0.2,
                                             recurrent_initializer='glorot_uniform')(encoder_embed)
    model = Model(encoder_input, [encoder_output, encoder_hid])

    return model


def build_decoder(decoder_input, hidden, encoder_output, vocab_size, embed_dim, unit_size):
    # shape (batch_size, 1, embed_dim) decoder
    decoder_embed = Embedding(vocab_size, embed_dim)(decoder_input)

    # shape (batch_size, encoder_units), (batch_size, encoder_seq_len, 1)
    context_vector, attention_weights = BahdanauAttention(unit_size, trainable=True)([hidden, encoder_output])

    # shape (batch_size, 1, embedding_dim + encoder_units)
    output = Concatenate(axis=-1)([tf.expand_dims(context_vector, 1), decoder_embed])

    # shape (batch_size, 1, decoder_units), (batch_size, decoder_units)
    output, decoder_hid, decoder_cell = LSTM(unit_size, return_sequences=True, return_state=True, dropout=0.2,
                         recurrent_initializer='glorot_uniform')(output)
    # (batch_size * 1, decoder_units)
    output = Reshape(target_shape=(-1, unit_size))(output)
    # shape (batch_size, vocab_size)
    output = Dense(vocab_size)(output)
    model = Model(inputs=[decoder_input, hidden, encoder_output], outputs=[output, decoder_hid])

    return model


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