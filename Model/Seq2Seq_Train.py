#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Graduate Project: Named Entity Recognition Task in Chinese Media
Author: Bin Wei
Description: This is part of my graduate project, which tries to explore some
            models targeting NER tasks in Chinese social media.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from collections import Counter
from Model.BiLSTM_CRF import calculate_metrics
from Model.Seq2Seq import Encoder, Decoder, train_step, predict
from Model.BiLSTM_CRF_Train import load_data_helper, store_labels, report_perfomence


def train_Seq2Seq():
    """
    This method trains the model and generates the predictions.
    """
    EMBED_DIM = 50
    HIDDEN_DIM = 32
    EPOCH = 20
    LEARNING_RATE = 0.01


    # Use the augmented training set to train the model.
    train_dataset, vocab_size, tag_size = load_data_helper("train", "seq2seq")
    model = BiLSTM_CRF(HIDDEN_DIM, vocab_size, tag_size, EMBED_DIM)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    for e in range(EPOCH):
        for i, (data_batch, label_batch) in enumerate(train_dataset):
            loss, logits, text_lens = train_step(model, data_batch, label_batch, optimizer)
            if (i + 1) % 10 == 0:
                print("Epoch:", e + 1, " Loss:", loss.numpy())

    # Generate the predictions on the following datasets and write them into corresponding pickle files.
    store_labels(model, train_dataset, "trainset")

    dev_dataset, _, _ = load_data_helper("dev", "seq2seq")
    store_labels(model, dev_dataset, "devset")

    test_dataset, _, _ = load_data_helper("test", "seq2seq")
    store_labels(model, test_dataset, "testset")

    # origin_train_dataset = load_data_helper("origin_train")
    # store_labels(model, origin_train_dataset, t)

if __name__ == "__main__":
    cond1 = os.path.exists("../Data/labels_testset_seq2seq.pkl")
    cond2 = os.path.exists("../Data/labels_trainset_seq2seq.pkl")
    cond3 = os.path.exists("../Data/labels_devset_seq2seq.pkl")
    if cond1 == cond2 == cond3 == True:
        report_perfomence("trainset_seq2seq")
        report_perfomence("devset_seq2seq")
        report_perfomence("testset_seq2seq")

    else:
        train_Seq2Seq()