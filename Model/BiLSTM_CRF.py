#! usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_addons as tf_ad

class BiLSTM_CRF(tf.keras.Model):
    def __init__(self, hidden_dim, vocab_size, tag_size, embed_dim):
        super(BiLSTM_CRF, self).__init__()
        self.num_hidden = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(tag_size)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(tag_size, tag_size)))
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
                                                    label_sequences, text_lens, transition_params=self.transition_params)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens


# @tf.function
def train_one_step(model, data, label, opt):
    with tf.GradientTape() as tape:
      logits, text_lens, log_likelihood = model(data, label,training=True)
      loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, logits, text_lens


def predict(model, labels, data):
    predictions = []
    logits, text_lens, log_likeliyhood = model(data, labels)
    for logit, text_len, label in zip(logits, text_lens, labels):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        viterbi_path = np.array(viterbi_path)
        predictions.append(viterbi_path)

    return predictions


def extract_label(text_lens, labels):
    real_labels = []
    for (text_len, label) in zip(text_lens, labels):
        real_label = np.array(label[:text_len])
        real_labels.append(real_label)
    return real_labels


def calculate_metrics(predictions, labels):
    pred_list = []
    label_list = []
    entity_list = []
    for i in range(len(predictions)):
        pred = predictions[i]
        label = labels[i]

        # temp_pred, temp_label = "", ""
        temp_pred, temp_label = [], []
        for k in range(len(label)):
            if label[k] != 16:
                temp_label.append(label[k])
                if k == len(label) - 1:
                    label_list.append(temp_label)
                # temp_label += str(label[k])
                # temp_pred += str(pred[k])
            else:
                # if len(temp_label) == len(temp_pred) == 0:
                if len(temp_label) == 0:
                    continue
                else:
                    # pred_list.append(temp_pred)
                    label_list.append(temp_label)
                    # temp_pred = temp_label = ""
                    temp_label = []

        for m in range(len(pred)):
            if pred[m] != 16:
                temp_pred.append(pred[m])
                if m == len(pred) - 1:
                    pred_list.append(temp_pred)
                # temp_label += str(label[k])
                # temp_pred += str(pred[k])
            else:
                # if len(temp_label) == len(temp_pred) == 0:
                if len(temp_pred) == 0:
                    continue
                else:
                    # pred_list.append(temp_pred)
                    pred_list.append(temp_pred)
                    # temp_pred = temp_label = ""

                    temp_pred = []


        temp_entity = []
        for s in range(len(label)):
            if label[s] != 16 and pred[s] != 16:
                if label[s] == pred[s]:
                    temp_entity.append(label[s])
                    if s == len(label) - 1:
                        entity_list.append(temp_entity)

            elif label[s] == pred[s] == 16:
                if len(temp_entity) == 0:
                    continue
                else:
                    entity_list.append(temp_entity)
                    temp_entity = []

    precision = len(entity_list) / len(pred_list)
    recall = len(entity_list) / len(label_list)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1
