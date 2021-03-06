#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Graduate Project: Named Entity Recognition Task in Chinese Media
Author: Bin Wei
Description: This is part of my graduate project, which tries to explore some
            models targeting NER tasks in Chinese social media.
"""

import pickle
import numpy as np
import regex as re
import tensorflow as tf

def read_file(file, data):
    """
    This method reads a file and return all necessary dictionaries and data.
    :param file: file to be read
    :return: all necessary dictionaries and data
    """
    tag_list = []
    char_list = []
    sentence_list = []
    sentence = []
    tags, tag = [], ""
    if data == "twitter":
        with open(file) as f:
            for line in f:
                if line == "\n":
                    sentence_list.append(sentence)
                    tags.append(tag.split())
                    sentence = []
                    tag = ""
                    continue
                else:
                    # Resolve the encoding problem.
                    if "\u200b" in line:
                        line = re.sub("\u200b", "O", line)
                    ls = line.split("\t")
                    char = ls[0]
                    sentence.append(char)
                    char_list.append(char)
                    tag_list.append(ls[-1].rstrip())
                    tag += ls[-1]

    else:
        with open(file) as f:
            for line in f:
                if line == "\n":
                    sentence_list.append(sentence)
                    tags.append(tag.split())
                    sentence = []
                    tag = ""
                    continue
                else:
                    # Resolve the encoding problem.
                    if "\u200b" in line:
                        line = re.sub("\u200b", "O", line)
                    ls = line.split("\t")
                    char = [char for char in ls[0]][0]
                    sentence.append(char)
                    char_list.append(char)
                    tag_list.append(ls[-1].rstrip())
                    tag += ls[-1]

    tag_list = []
    for item in tags:
        tag_list.extend(item)
    tag_list = sorted(set(tag_list))

    # Clean the items in the tag_list and construct a dictionary out of the list.
    for i in range(len(tag_list)):
        s = tag_list[i].rstrip("\n")
        tag_list[i] = s
    tag_dic = {n:m for m, n in enumerate(tag_list)}

    char_list = sorted(set(char_list))
    char_dic = {v:k for k, v in enumerate(char_list)}
    sentence_list, tags = clean_data(sentence_list, tags)

    return tag_dic, char_dic, sentence_list, tags


def clean_data(data, label):
    data_ls = []
    label_ls = []
    for i in range(len(data)):
        if len(data[i]) > 0 :
            data_ls.append(data[i])
            label_ls.append(label[i])

    return data_ls, label_ls


def save_data(file, tag_dic, char_dic, sentence_list, tags):
    """
    This method writes all possibly relevant dictionaries and data into pickle files.
    :param file: the file to be read
    :param tag_dic: the tag dictionary used for converting data
    :param char_dic: the character dictionary used for converting data
    :param sentence_list: all the sentences in a file
    :param tags: all the tags corresponding to the sentences
    :return: none
    """
    data, label = [], []
    # Use the character dictionary and tag dictionary to convert the data.
    for s in range(len(sentence_list)):
        sentence = [char for char in sentence_list[s]]
        tag = [t for t in tags[s]]

        d = []
        for char in sentence:
            for k in char_dic.keys():
                if char == k:
                    d.append(char_dic[k])
        data.append(d)

        l = []
        for t in tag:
            for k in tag_dic.keys():
                if t == k:
                    l.append(tag_dic[k])
        label.append(l)

    # Set up the name for the corresponding pickle file.
    with open(file, "wb") as pkl:
        pickle.dump(tag_dic, pkl)
        pickle.dump(char_dic, pkl)
        pickle.dump(sentence_list, pkl)
        pickle.dump(tags, pkl)
        pickle.dump(data, pkl)
        pickle.dump(label, pkl)


def format_data(data, label, tag_dic):
    """
    This method finds the longest label and uses this to pad in both data and label.
    :param data: data to be processed
    :param label: label to be processed
    :return: label and data with paddings
    """
    data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=256, padding="post", value=0)
    label = tf.keras.preprocessing.sequence.pad_sequences(label, maxlen=256, padding="post", value=tag_dic["O"])
    return data, label


def write_files(keyword):
    """
    This method writes the converted data from different datasets into corresponding pickle files.
    :param keyword: keyword of options
    """
    if keyword == "weibo":
        tag_dic, char_dic, sentence_list, tags = read_file("../Data/Chinese_Weibo_NER_Corpus.train", keyword)
        _, _, sentence_list_b, tags_b = read_file("../Data/weiboNER_2nd_conll.dev", keyword)
        _, _, sentence_list_c, tags_c = read_file("../Data/weiboNER_2nd_conll.test", keyword)
        save_data("../PickleFiles/Chinese_Weibo_NER_Corpus_train.pkl", tag_dic, char_dic, sentence_list, tags)
        save_data("../PickleFiles/Chinese_Weibo_NER_Corpus_dev.pkl", tag_dic, char_dic, sentence_list_b, tags_b)
        save_data("../PickleFiles/Chinese_Weibo_NER_Corpus_test.pkl", tag_dic, char_dic, sentence_list_c, tags_c)

    elif keyword == "weibo_origin":
        tag_dic, char_dic, sentence_list, tags = read_file("../Data/weiboNER_2nd_conll.train", keyword)
        _, _, sentence_list_b, tags_b = read_file("../Data/weiboNER_2nd_conll.dev", keyword)
        _, _, sentence_list_c, tags_c = read_file("../Data/weiboNER_2nd_conll.test", keyword)
        save_data("../PickleFiles/Chinese_Weibo_NER_Corpus_train_origin.pkl", tag_dic, char_dic, sentence_list, tags)
        save_data("../PickleFiles/Chinese_Weibo_NER_Corpus_dev_origin.pkl", tag_dic, char_dic, sentence_list_b, tags_b)
        save_data("../PickleFiles/Chinese_Weibo_NER_Corpus_test_origin.pkl", tag_dic, char_dic, sentence_list_c, tags_c)

    elif keyword == "msra":
        tag_dic, char_dic, sentence_list, tags = read_file("../Data/Chinese_MSRA_NER_Corpus.train", keyword)
        _, _, sentence_list_a, tags_a = read_file("../Data/Chinese_MSRA_NER_Corpus.test", keyword)
        save_data("../PickleFiles/Chinese_MSRA_NER_Corpus_train.pkl", tag_dic, char_dic, sentence_list, tags)
        save_data("../PickleFiles/Chinese_MSRA_NER_Corpus_test.pkl", tag_dic, char_dic, sentence_list_a, tags_a)

    else:
        tag_dic, char_dic, sentence_list, tags = read_file("../Data/English_Twitter_NER_Corpus.train", keyword)
        _, _, sentence_list_a, tags_a = read_file("../Data/English_Twitter_NER_Corpus.test", keyword)
        save_data("../PickleFiles/English_Twitter_NER_Corpus_train.pkl", tag_dic, char_dic, sentence_list, tags)
        save_data("../PickleFiles/English_Twitter_NER_Corpus_test.pkl", tag_dic, char_dic, sentence_list_a, tags_a)


if __name__ == "__main__":
    write_files("weibo")
    write_files("weibo_origin")
    write_files("msra")
    write_files("twitter")
    print("Writing pickle files done.")

