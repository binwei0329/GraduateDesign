#! usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import pickle
import numpy as np
import tensorflow as tf
from collections import Counter
from Model.Preprocess import format_data, load_char_embeddings
from Model.BiLSTM_CRF import BiLSTM_CRF, train_one_step, predict, calculate_metrics

def load_data(file):
    with open(file, "rb") as f:
        tag_dic = pickle.load(f)
        char_dic = pickle.load(f)
        word_dic = pickle.load(f)
        sentence_list = pickle.load(f)
        tags = pickle.load(f)
        data = pickle.load(f)
        label = pickle.load(f)

    data, label = format_data(data, label)
    vocab_size = len(char_dic)
    tag_size = len(tag_dic)

    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(len(data)).batch(64, drop_remainder=True)
    return dataset, vocab_size, tag_size


def load_data_helper(op):
    if op == "train":
        with open("../Data/weiboNER_2nd_conll.train.pkl", "rb") as file:
            _ = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            data = pickle.load(file)
            label = pickle.load(file)
        length = len(data)

        with open("../Data/weiboNER_Corpus.train.pkl", "rb") as file_train:
            tag_dic = pickle.load(file_train)
            char_dic = pickle.load(file_train)
            word_dic = pickle.load(file_train)
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

        data_train, label_train = format_data(data_train, label_train)
        vocab_size = len(char_dic)
        tag_size = len(tag_dic)

        dataset = tf.data.Dataset.from_tensor_slices((data_train, label_train))
        dataset = dataset.shuffle(len(data_train)).batch(64, drop_remainder=True)

    elif op == "dev":
        file = "../Data/weiboNER_2nd_conll.dev.pkl"
        dataset, vocab_size, tag_size = load_data(file)

    else:
        file = "../Data/weiboNER_2nd_conll.test.pkl"
        dataset, vocab_size, tag_size = load_data(file)

    # else:
    #     file = "../Data/weiboNER_2nd_conll.train.pkl"
    #     dataset, vocab_size, tag_size = load_data(file)

    return dataset, vocab_size, tag_size


def store_labels(model, dataset, op):
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

    filename = "../Data/labels_" + op + ".pkl"
    if os.path.exists(filename) == False:
        with open(filename, "wb") as file:
            pickle.dump(pred_labels, file)
            pickle.dump(gold_labels, file)


def traing_BiLSTM_CRF():
    EMBED_DIM = 50
    HIDDEN_DIM = 32
    EPOCH = 20
    LEARNING_RATE = 0.01

    char_embed_dict = load_char_embeddings("../Data/vec.txt")

    train_dataset, vocab_size, tag_size = load_data_helper("train")
    model = BiLSTM_CRF(HIDDEN_DIM, vocab_size, tag_size, EMBED_DIM)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    for e in range(EPOCH):
        for i, (data_batch, label_batch) in enumerate(train_dataset):
            loss, logits, text_lens = train_one_step(model, data_batch, label_batch, optimizer)
            if (i + 1) % 10 == 0:
                print("Epoch:", e + 1, " Loss:", loss.numpy())

    store_labels(model, train_dataset, "trainset")

    dev_dataset, _, _ = load_data_helper("dev")
    store_labels(model, dev_dataset, "devset")

    test_dataset, _, _ = load_data_helper("test")
    store_labels(model, test_dataset, "testset")

    # origin_train_dataset = load_data_helper("origin_train")
    # store_labels(model, origin_train_dataset, t)

def report_perfomence(arg):
    filename = "../Data/labels_" + arg + ".pkl"

    with open(filename, "rb") as file:
        prediction = pickle.load(file)
        gold = pickle.load(file)

    precision, recall, f1 = calculate_metrics(prediction, gold)
    print(arg, "performance precision: %.2f recall: %.2f f1score: %.2f" %(precision*100, recall*100, f1*100))

if __name__ == "__main__":
    cond1 = os.path.exists("../Data/labels_testset.pkl")
    cond2 = os.path.exists("../Data/labels_trainset.pkl")
    cond3 = os.path.exists("../Data/labels_devset.pkl")
    if cond1 == cond2 == cond3 == True:
        report_perfomence("trainset")
        report_perfomence("devset")
        report_perfomence("testset")

    else:
        traing_BiLSTM_CRF()
