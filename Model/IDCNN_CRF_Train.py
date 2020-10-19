#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Graduate Project: Named Entity Recognition Task in Chinese Media
Author: Bin Wei
Description: This is part of my graduate project, which tries to explore some
            models targeting NER tasks in Chinese social media.
"""

import os
import time
import pickle
import numpy as np
import tensorflow as tf
from Model.Statistics import extract_labels
from Model.IDCNN_CRF import IDCNN_CRF, train_one_step, predict
from Model.BiLSTM_CRF_Train import load_data, load_data_helper

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


def save_labels(model, dataset, name):
    """
    The method uses the trained model to generate predicted labels and
    writes both predicted labels and gold labels into corresponding pickle files.
    :param model: trained model
    :param dataset: dataset on which the model makes predictions
    :param op: name to choose
    """
    pred_labels = []
    gold_labels = []
    for i, (data, label) in enumerate(dataset):
        gold_labels.extend(np.array(label))
        predictions = predict(model, label, data)
        pred_labels.extend(predictions)

    for s in range(len(pred_labels)):
        pred_len = len(pred_labels[s])
        gold_label = gold_labels[s]
        effective_part = gold_label[:pred_len]
        gold_labels[s] = effective_part

    filename = "../PickleFiles/Labels_Idcnn_crf_" + name + ".pkl"
    if os.path.exists(filename) == False:
        with open(filename, "wb") as file:
            pickle.dump(pred_labels, file)
            pickle.dump(gold_labels, file)


def test_model(model, data_op, trainset):
    if data_op == "weibo":
        save_labels(model, trainset, "weibo_train")
        dev_dataset, _, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_dev.pkl", batch_size=64)
        save_labels(model, dev_dataset, "weibo_dev")

        test_dataset, _, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_test.pkl", batch_size=64)
        save_labels(model, test_dataset, "weibo_test")

    else:
        save_labels(model, trainset, "weibo_train_origin")
        dev_dataset, _, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_dev_origin.pkl", batch_size=64)
        save_labels(model, dev_dataset, "weibo_dev_origin")

        test_dataset, _, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_test_origin.pkl", batch_size=64)
        save_labels(model, test_dataset, "weibo_test_origin")


def report_perfomence(arg, tag_dic):
    """
    This method gives relevant evaluations on the performance.
    :param arg: dataset to choose
    :param tag_dic: tag dictionary
    :return: precision, recall, f1
    """
    filename = "../PickleFiles/Labels_Idcnn_crf_" + arg + ".pkl"

    with open(filename, "rb") as file:
        prediction = pickle.load(file)
        gold = pickle.load(file)

    entity_ls, pred_ls, label_ls = extract_labels(prediction, gold, tag_dic)

    precision = len(entity_ls) / len(pred_ls)
    recall = len(entity_ls) / len(label_ls)
    f1 = 2 * precision * recall / (precision + recall)

    precision = format(precision, ".4f")
    recall = format(recall, ".4f")
    f1 = format(f1, ".4f")

    return precision, recall, f1


if __name__ == "__main__":
    weibo_train, vocab_size, tag_size, tag_dic = load_data_helper(64)
    model = train_IDCNN_CRF(weibo_train, vocab_size, tag_size, 10)
    test_model(model, "weibo", weibo_train)

    start = time.asctime(time.localtime(time.time()))
    print(start)
    weibo_train_origin, vocab_size_o, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_train_origin.pkl", batch_size=64)
    model_o = train_IDCNN_CRF(weibo_train_origin, vocab_size_o, tag_size, 10)
    end = time.asctime(time.localtime(time.time()))
    print(end)
    print(model_o.summary())
    test_model(model_o, "weibo_origin", weibo_train_origin)
    print("Model trained and predictions given.")

