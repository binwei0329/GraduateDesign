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
    # word_list = []
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
                        line = re.sub("\u200b", "X", line)
                    ls = line.split("\t")
                    # print(ls)

                    # print(ls[-1].rstrip())
                    # char = [char for char in ls[0]][0]
                    char = ls[0]
                    sentence.append(char)
                    char_list.append(char)
                    # word_list.append(ls[0])
                    # print(ls[-1].rstrip())
                    tag_list.append(ls[-1].rstrip())
                    tag += ls[-1]

    elif data == "msra" or "weibo":
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
                        line = re.sub("\u200b", "X", line)
                    ls = line.split("\t")
                    # print(ls[-1].rstrip())
                    char = [char for char in ls[0]][0]
                    sentence.append(char)
                    char_list.append(char)
                    # word_list.append(ls[0])
                    # print(ls[-1].rstrip())
                    tag_list.append(ls[-1].rstrip())
                    tag += ls[-1]
    # if data == "twitter":
    #     with open(file) as f:
    #         for line in f:
    #             if line == "\n":
    #                 sentence_list.append(sentence)
    #                 tags.append(tag.split())
    #                 sentence = []
    #                 tag = ""
    #                 continue
    #             else:
    #                 # Resolve the encoding problem.
    #                 if "\u200b" in line:
    #                     line = re.sub("\u200b", "X", line)
    #                 ls = line.split("\t")
    #                 print(ls)
    #                 # print(ls[-1].rstrip())
    #                 # char = [char for char in ls[0]][0]
    #                 # sentence.append(char)
    #                 # char_list.append(char)
    #                 # word_list.append(ls[0])
    #                 # print(ls[-1].rstrip())
    #                 tag_list.append(ls[-1].rstrip())
    #                 tag += ls[-1]


    # if data == "weibo":
    #     # Exclude the last tag symbol, "X".
    #     tag_list = tag_list[:-1]
    # # elif data == "msra":
    # elif data == "msra" or "twitter":
    #     tag_list = tag_list[1:]


    tag_list = sorted(set(tag_list))

    if data == "weibo":
        # Exclude the last tag symbol, "X".
        tag_list = tag_list[:-1]
    # elif data == "msra":

    # elif data == "msra" or "twitter":
    else:
        tag_list = tag_list[1:]

    # Clean the items in the tag_list and construct a dictionary out of the list.
    for i in range(len(tag_list)):
        s = tag_list[i].rstrip("\n")
        tag_list[i] = s
    tag_dic = {n:m for m, n in enumerate(tag_list)}

    word = []
    temp = ""
    char = []

    # for i in range(len(word_list)):
    #     char_ls = [char for char in word_list[i]]
    #     char.extend(char_ls)
    #     temp += char_ls[0]
    #
    #     # Get the segmented words.
    #     if word_list[i].endswith("0") and i > 0:
    #         word.append(temp[:-1])
    #         temp = temp[-1] + ""

    # word = sorted(set(word))
    # word_dic = {v:k for k, v in enumerate(word)}
    char_list = sorted(set(char_list))
    char_dic = {v:k for k, v in enumerate(char_list)}
    # return tag_dic, char_dic, word_dic, sentence_list, tags
    return tag_dic, char_dic, sentence_list, tags


def load_char_embeddings(file):
    """
    This method extracts the character embeddings from file.
    :param file: the embedding file
    :return: character embeddings
    """
    f = open(file,'r')
    char_embed_dict = {}
    for line in f:
        splitLines = line.split()
        char = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        char_embed_dict[char] = wordEmbedding
    return char_embed_dict


def convert_data(file, tag_dic, char_dic, word_dic, sentence_list, tags):
    """
    This method writes all possibly relevant dictionaries and data into pickle files.
    :param file: the file to be read
    :param tag_dic: the tag dictionary used for converting data
    :param char_dic: the character dictionary used for converting data
    :param word_dic: the word dictionary used for converting data
    :param sentence_list: all the sentences in a file
    :param tags: all the tags corresponding to the sentences
    :return: none
    """
    data, label = [], []
    # Use the character dictionary and tag dictionary to convert the data.
    for s in range(len(sentence_list)):
        d, l = [], []
        sentence = [char for char in sentence_list[s]]
        tag = [t for t in tags[s]]
        for char in sentence:
            for k in char_dic.keys():
                if char == k:
                    d.append(char_dic[k])
        data.append(d)

        for t in tag:
            for k in tag_dic.keys():
                if t == k:
                    l.append(tag_dic[k])
        label.append(l)

    # Set up the name for the corresponding pickle file.
    name = str(file)[str(file).index("weibo"):]
    file_name = "../Data/" + name + "2.pkl"
    with open(file_name, "wb") as pkl:
        pickle.dump(tag_dic, pkl)
        pickle.dump(char_dic, pkl)
        pickle.dump(word_dic, pkl)
        pickle.dump(sentence_list, pkl)
        pickle.dump(tags, pkl)
        pickle.dump(data, pkl)
        pickle.dump(label, pkl)


def format_data(data, label):
    """
    This method finds the longest label and uses this to pad in both data and label.
    :param data: data to be processed
    :param label: label to be processed
    :return: label and data with paddings
    """
    data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=256, padding="post", value=0)
    label = tf.keras.preprocessing.sequence.pad_sequences(label, maxlen=256, padding="post", value=16)
    return data, label


def main():
    """
    This method writes the converted data from different datasets into corresponding pickle files.
    """
    # tag_dic, char_dic, word_dic, sentence_list, tags = read_file("../Data/weiboNER_Corpus.train")
    # _, _, _, sentence_list_a, tags_a = read_file("../Data/weiboNER_2nd_conll.train")
    # _, _, _, sentence_list_b, tags_b = read_file("../Data/weiboNER_2nd_conll.dev")
    # _, _, _, sentence_list_c, tags_c = read_file("../Data/weiboNER_2nd_conll.test")

    # convert_data("../Data/weiboNER_Corpus.train", tag_dic, char_dic, word_dic, sentence_list, tags)
    # convert_data("../Data/weiboNER_2nd_conll.train", tag_dic, char_dic, word_dic, sentence_list_a, tags_a)
    # convert_data("../Data/weiboNER_2nd_conll.dev", tag_dic, char_dic, word_dic, sentence_list_b, tags_b)
    # convert_data("../Data/weiboNER_2nd_conll.test", tag_dic, char_dic, word_dic, sentence_list_c, tags_c)


if __name__ == "__main__":
    # main()
    tag_dic, char_dic, sentence_list, tags = read_file("../Data/Chinese_MSRA_NER_Corpus.train", "msra")
    # tag_dic, char_dic, sentence_list, tags = read_file("../Data/Chinese_Weibo_NER_Corpus.train", "weibo")
    #
    # tag_dic, char_dic, sentence_list, tags = read_file("../Data/English_Twitter_NER_Corpus.train", "twitter")

    print(tag_dic, "\n", sentence_list[0], "\n",tags[0])
    print(char_dic)