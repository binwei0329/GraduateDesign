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


def load_data_helper(data_op, batch_size):
    """
    This method returns the formatted dataset, vocab size and tag size of different files.
    :param op: the options
    :return: formatted dataset, vocab size and tag size
    """
    # If the option is "train", we apply undersampling and oversampling technique to get
    # bigger datasets.
    if data_op == "train":
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

    elif data_op == "dev":
        file = "../PickleFiles/Chinese_Weibo_NER_Corpus_dev.pkl"
        dataset, vocab_size, tag_size = load_data(file, batch_size)

    elif data_op == "test":
        file = "../PickleFiles/Chinese_MSRA_NER_Corpus_test.pkl"
        dataset, vocab_size, tag_size = load_data(file, batch_size)

    else:
        file = "../PickleFiles/Chinese_Weibo_NER_Corpus_original_train.pkl"
        dataset, vocab_size, tag_size = load_data(file, batch_size)

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


def train_BiLSTM_CRF(batch_op):
    """
    This method trains the model and generates the predictions.
    """
    EMBED_DIM = 64
    UNIT_NUM = 64
    EPOCH = 20
    LEARNING_RATE = 0.005
    BATCH_SIZE = {"weibo":64, "msra":10240, "twitter":2048}
    char_embed_dict = load_char_embeddings("../Data/vec.txt")

    # Use the augmented training set to train the model.
    train_dataset, vocab_size, tag_size = load_data_helper("train", BATCH_SIZE[batch_op])
    model = BiLSTM_CRF(UNIT_NUM, vocab_size, tag_size, EMBED_DIM)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    for e in range(EPOCH):
        for i, (data_batch, label_batch) in enumerate(train_dataset):
            print(label_batch)
            loss, logits, text_lens = train_one_step(model, data_batch, label_batch, optimizer)
            print(logits)
            if (i + 1) % 10 == 0:
                print("Epoch:", e + 1, "Batch:", i + 1, "Loss:", loss.numpy())

    # Generate the predictions on the following datasets and write them into corresponding pickle files.
    save_labels(model, train_dataset, "trainset")

    dev_dataset, _, _ = load_data_helper("dev", BATCH_SIZE[batch_op])
    save_labels(model, dev_dataset, "devset")

    test_dataset, _, _ = load_data_helper("test", BATCH_SIZE[batch_op])
    save_labels(model, test_dataset, "testset")

    # origin_train_dataset = load_data_helper("origin_train")
    # store_labels(model, origin_train_dataset, t)

def report_perfomence(arg):
    """
    This method gives relevant evaluations on the performance.
    :param arg: dataset to choose
    """
    filename = "../Data/bilstm_crf_labels_" + arg + ".pkl"

    with open(filename, "rb") as file:
        prediction = pickle.load(file)
        gold = pickle.load(file)

    precision, recall, f1 = calculate_metrics(prediction, gold)
    print(arg, "performance precision: %.2f recall: %.2f f1score: %.2f" %(precision*100, recall*100, f1*100))


if __name__ == "__main__":
    # cond1 = os.path.exists("../Data/bilstm_crf_labels_testset.pkl")
    # cond2 = os.path.exists("../Data/bilstm_crf_labels_trainset.pkl")
    # cond3 = os.path.exists("../Data/bilstm_crf_labels_devset.pkl")
    # if cond1 == cond2 == cond3 == True:
    #     report_perfomence("trainset")
    #     report_perfomence("devset")
    #     report_perfomence("testset")
    #
    # else:
        train_BiLSTM_CRF()
