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
from Model.Preprocess import format_data, clean_data
from Model.BiLSTM_CRF import BiLSTM_CRF, train_one_step, predict

def load_data(file, batch_size):
    """
    This method reads the pickle files and format them.
    :param file: file to be read
    :param batch_size: the size of a batch
    :return: the formatted dataset, tag size, vocab size and tag_dic
    """
    with open(file, "rb") as f:
        tag_dic = pickle.load(f)
        char_dic = pickle.load(f)
        sentence_list = pickle.load(f)
        tags = pickle.load(f)
        data = pickle.load(f)
        label = pickle.load(f)

    data, label = clean_data(data, label)
    vocab_size = len(char_dic)
    tag_size = len(tag_dic)
    data, label = format_data(data, label, tag_dic)
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(len(data)).batch(batch_size, drop_remainder=True)
    return dataset, vocab_size, tag_size, tag_dic


def load_data_helper(batch_size):
    """
    This method returns the formatted dataset, vocab size, tag size and tag_dic of different files.
    :param batch_size: the size of a batch
    :return: formatted dataset, vocab size, tag size and tag_dic
    """
    # If the option is "train", we apply undersampling and oversampling technique to get bigger datasets.
    with open("../PickleFiles/Chinese_Weibo_NER_Corpus_train_origin.pkl", "rb") as file:
        _ = pickle.load(file)
        _ = pickle.load(file)
        _ = pickle.load(file)
        _ = pickle.load(file)
        data = pickle.load(file)
        label = pickle.load(file)

    length = len(data)
    with open("../PickleFiles/Chinese_Weibo_NER_Corpus_train.pkl", "rb") as file_train:
        tag_dic = pickle.load(file_train)
        char_dic = pickle.load(file_train)
        sentence_list = pickle.load(file_train)
        tags = pickle.load(file_train)
        data_train = pickle.load(file_train)
        label_train = pickle.load(file_train)

    data_duplicate = data_train[length:]
    label_duplicate = label_train[length:]

    # Oversampling the named entities.
    for i in range(3):
        data_train.extend(data_duplicate)
        label_train.extend(label_duplicate)

    # Undersampling the data without named entities.
    for i in range(length):
        label = label_train[i]
        dic = Counter(label)
        if dic[16] == len(label):
            del label_train[i]
            del data_train[i]

    data_train, label_train = format_data(data_train, label_train, tag_dic)
    vocab_size = len(char_dic)
    tag_size = len(tag_dic)

    dataset = tf.data.Dataset.from_tensor_slices((data_train, label_train))
    dataset = dataset.shuffle(len(data_train)).batch(batch_size, drop_remainder=True)
    return dataset, vocab_size, tag_size, tag_dic


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

    # filename = "Labels_Bilstm_crf_" + name + ".pkl"

    filename = "../PickleFiles/Labels_Bilstm_crf_" + name + ".pkl"
    if os.path.exists(filename) == False:
        with open(filename, "wb") as file:
            pickle.dump(pred_labels, file)
            pickle.dump(gold_labels, file)


def train_BiLSTM_CRF(dataset, vocab_size, tag_size, epoch):
    """
    This method trains a model and generates the predictions.
    :param dataset: a dataset for training
    :param vocab_size: size of the vocabulary
    :param tag size: number of tags
    :param epoch: number of epoches
    """
    EMBED_DIM = 32
    UNIT_NUM = 64
    LEARNING_RATE = 0.005
    model = BiLSTM_CRF(UNIT_NUM, vocab_size, tag_size, EMBED_DIM)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    for e in range(epoch):
        for i, (data_batch, label_batch) in enumerate(dataset):
            loss, logits, text_lens = train_one_step(model, data_batch, label_batch, optimizer)
            if (i + 1) % 10 == 0:
                print("Epoch:", e + 1, "Batch:", i + 1, "Loss:", loss.numpy())

    return model


def test_model(model, data_op, trainset):
    """
    This method tests the trained model on different data-sets.
    :param model: a trained model
    :param data_op: option of data-set
    :param trainset: trainning set
    :return: none
    """
    if data_op == "msra":
        save_labels(model, trainset, "msra_train")
        test_dataset, _, _, _ = load_data("../PickleFiles/Chinese_MSRA_NER_Corpus_test.pkl", batch_size=512)
        save_labels(model, test_dataset, "msra_test")

    elif data_op == "twitter":
        save_labels(model, trainset, "twitter_train")
        test_dataset, _, _, _ = load_data("../PickleFiles/English_Twitter_NER_Corpus_test.pkl", batch_size=256)
        save_labels(model, test_dataset, "twitter_test")

    elif data_op == "weibo":
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


if __name__ == "__main__":
    weibo_train, vocab_size, tag_size, _ = load_data_helper(batch_size=64)
    weibo_train_origin, _, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_train_origin.pkl", batch_size=64)
    msra_train, vocab_size_m, tag_size_m, _ = load_data("../PickleFiles/Chinese_MSRA_NER_Corpus_train.pkl", batch_size=512)
    twitter_train, vocab_size_t, tag_size_t, _ = load_data("../PickleFiles/English_Twitter_NER_Corpus_train.pkl", batch_size=256)

    model_m = train_BiLSTM_CRF(msra_train, vocab_size_m, tag_size_m, epoch=20)
    test_model(model_m, "msra", msra_train)

    model_t = train_BiLSTM_CRF(twitter_train, vocab_size_t, tag_size_t, epoch=20)
    test_model(model_t, "twitter", twitter_train)

    model_o = train_BiLSTM_CRF(weibo_train_origin, vocab_size, tag_size, epoch=20)
    test_model(model_o, "weibo_origin", weibo_train_origin)

    model_w = train_BiLSTM_CRF(weibo_train, vocab_size, tag_size, epoch=20)
    test_model(model_w, "weibo", weibo_train)



