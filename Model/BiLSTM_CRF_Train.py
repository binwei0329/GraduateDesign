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
from Model.Preprocess import format_data, load_char_embeddings
from Model.BiLSTM_CRF import BiLSTM_CRF, train_one_step, predict, calculate_metrics

def load_data(file, batch_size):
    """
    This method reads the pickle files and format them.
    :param file: file to be read
    :return: the formatted dataset, tag size and vocab size
    """
    with open(file, "rb") as f:
        tag_dic = pickle.load(f)
        char_dic = pickle.load(f)
        sentence_list = pickle.load(f)
        tags = pickle.load(f)
        data = pickle.load(f)
        label = pickle.load(f)
    data, label = format_data(data, label, tag_dic)
    vocab_size = len(char_dic)
    tag_size = len(tag_dic)

    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(len(data)).batch(batch_size, drop_remainder=True)
    return dataset, vocab_size, tag_size


def load_data_helper(batch_size):
    # data_op,
    """
    This method returns the formatted dataset, vocab size and tag size of different files.
    :param op: the options
    :return: formatted dataset, vocab size and tag size
    """
    # If the option is "train", we apply undersampling and oversampling technique to get
    # bigger datasets.
    # if data_op == "train":
    with open("../PickleFiles/Chinese_Weibo_NER_Corpus_original_train.pkl", "rb") as file:
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
    return dataset, vocab_size, tag_size


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

    filename = "../PickleFiles/Labels_Bilstm_crf_" + name + ".pkl"
    if os.path.exists(filename) == False:
        with open(filename, "wb") as file:
            pickle.dump(pred_labels, file)
            pickle.dump(gold_labels, file)


def train_BiLSTM_CRF(dataset, vocab_size, tag_size, epoch):
    """
    This method trains the model and generates the predictions.
    """
    EMBED_DIM = 64
    UNIT_NUM = 64
    LEARNING_RATE = 0.005
    model = BiLSTM_CRF(UNIT_NUM, vocab_size, tag_size, EMBED_DIM)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    # Use the augmented training set to train the model.
    for e in range(epoch):
        for i, (data_batch, label_batch) in enumerate(dataset):
            loss, logits, text_lens = train_one_step(model, data_batch, label_batch, optimizer)
            if (i + 1) % 10 == 0:
                print("Epoch:", e + 1, "Batch:", i + 1, "Loss:", loss.numpy())

    return model


def report_perfomence(arg):
    """
    This method gives relevant evaluations on the performance.
    :param arg: dataset to choose
    """
    filename = "../PickleFiles/Labels_Bilstm_crf_" + arg + ".pkl"

    with open(filename, "rb") as file:
        prediction = pickle.load(file)
        gold = pickle.load(file)

    precision, recall, f1 = calculate_metrics(prediction, gold)
    print(arg, "performance precision: %.2f recall: %.2f f1score: %.2f" %(precision*100, recall*100, f1*100))


def test_model(model, data_op, trainset):
    if data_op == "msra":
        save_labels(model, trainset, "msra_train")
        test_dataset, _, _ = load_data("../PickleFiles/Chinese_MSRA_NER_Corpus_test.pkl", batch_size=512)
        save_labels(model, test_dataset, "msra_test")

    elif data_op == "twitter":
        save_labels(model, trainset, "twitter_train")
        test_dataset, _, _ = load_data("../PickleFiles/English_Twitter_NER_Corpus_train.pkl", batch_size=128)
        save_labels(model, test_dataset, "twitter_test")

    else:
        if data_op == "weibo":
            save_labels(model, trainset, "weibo_train")
        else:
            save_labels(model, trainset, "weibo_train_origin")

        dev_dataset, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_dev.pkl", batch_size=64)
        save_labels(model, dev_dataset, "weibo_dev")

        test_dataset, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_test.pkl", batch_size=64)
        save_labels(model, test_dataset, "weibo_test")


if __name__ == "__main__":
    weibo_train, vocab_size, tag_size = load_data_helper(batch_size=64)
    weibo_train_origin, _, _ = load_data("../PickleFiles/Chinese_Weibo_NER_Corpus_original_train.pkl", batch_size=64)
    msra_train, vocab_size_m, tag_size_m = load_data("../PickleFiles/Chinese_MSRA_NER_Corpus_train.pkl", batch_size=512)
    twitter_train, vocab_size_t, tag_size_t = load_data("../PickleFiles/English_Twitter_NER_Corpus_train.pkl", batch_size=128)

    model_o = train_BiLSTM_CRF(weibo_train_origin, vocab_size, tag_size, epoch=20)
    test_model(model_o, "weibo_origin", weibo_train_origin)

    model_w = train_BiLSTM_CRF(weibo_train, vocab_size, tag_size, epoch=20)
    test_model(model_w, "weibo", weibo_train)

    model_m = train_BiLSTM_CRF(msra_train, vocab_size_m, tag_size_m, epoch=10)
    test_model(model_m, "msra", msra_train)

    model_t = train_BiLSTM_CRF(twitter_train, vocab_size_t, tag_size_t, epoch=10)
    test_model(model_t, "twitter", twitter_train)


