#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Graduate Project: Named Entity Recognition Task in Chinese Media
Author: Bin Wei
Description: This is part of my graduate project, which tries to explore some
            models targeting NER tasks in Chinese social media.
"""

import os
import tensorflow as tf
from Model.BiLSTM_CRF_Train import load_data, load_data_helper, save_labels
from Model.IDCNN_CRF import IDCNN_CRF, train_one_step

def train_IDCNN_CRF(dataset, vocab_size, tag_size, epoch):
    """
    This method trains a model and generates the predictions.
    :param dataset: a dataset for training
    :param vocab_size: size of the vocabulary
    :param tag size: number of tags
    :param epoch: number of epoches
    """
    EMBED_DIM = 32
    LEARNING_RATE = 0.005
    model = IDCNN_CRF(vocab_size, tag_size, EMBED_DIM)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    for e in range(epoch):
        for i, (data_batch, label_batch) in enumerate(dataset):
            loss, logits, text_lens = train_one_step(model, data_batch, label_batch, optimizer)
            if (i + 1) % 10 == 0:
                print("Epoch:", e + 1, "Batch:", i + 1, "Loss:", loss.numpy())

    return model


if __name__ == "__main__":
    dataset, vocab_size, tag_size, tag_dic = load_data_helper(64)
    train_IDCNN_CRF(dataset, vocab_size, tag_size, 20)
