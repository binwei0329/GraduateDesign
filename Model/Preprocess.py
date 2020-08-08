#! usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import regex as re
import tensorflow as tf


def read_file(file):
    tag_list = []
    word_list = []
    sentence_list = []
    sentence = []
    tags, tag = [], ""

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
                sentence.append([char for char in ls[0]][0])
                word_list.append(ls[0])
                tag_list.append(ls[-1])
                tag += ls[-1]

    tag_list = sorted(set(tag_list))

    # Exclude the last tag symbol, "X".
    tag_list = tag_list[:-1]

    # Clean the items in the tag_list and construct a dictionary out of the list.
    for i in range(len(tag_list)):
        s = tag_list[i].rstrip("\n")
        tag_list[i] = s
    tag_dic = {n:m for m, n in enumerate(tag_list)}
    # for n in tag_dic.keys():
    #     m = tag_dic[n] + 1
    #     tag_dic[n] = m

    word = []
    temp = ""
    char = []
    for i in range(len(word_list)):
        char_ls = [char for char in word_list[i]]
        char.extend(char_ls)
        temp += char_ls[0]

        # Get the segmented words.
        if word_list[i].endswith("0") and i > 0:
            word.append(temp[:-1])
            temp = temp[-1] + ""

    word = sorted(set(word))
    word_dic = {v:k for k, v in enumerate(word)}
    char = sorted(set(char))
    char_dic = {v:k for k, v in enumerate(char)}

    return tag_dic, char_dic, word_dic, sentence_list, tags

# tag_dic, char_dic, word_dic, sentence_list, tags = read_file("../Data/Weibo_NER_Corpus.train")
# print(tag_dic)


def load_char_embeddings(file):
    f = open(file,'r')
    char_embed_dict = {}
    for line in f:
        splitLines = line.split()
        char = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        char_embed_dict[char] = wordEmbedding
    return char_embed_dict


def convert_data(file, tag_dic, char_dic, word_dic, sentence_list, tags):
    name = str(file)[str(file).index("weibo"):]

    data, label = [], []
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
    file_name = "../Data/" + name + ".pkl"
    with open(file_name, "wb") as pkl:
        pickle.dump(tag_dic, pkl)
        pickle.dump(char_dic, pkl)
        pickle.dump(word_dic, pkl)
        pickle.dump(sentence_list, pkl)
        pickle.dump(tags, pkl)
        pickle.dump(data, pkl)
        pickle.dump(label, pkl)


def format_data(data, label):
    length = []
    for da in data:
        length.append(len(da))
    # avg_len = length // len(data)
    max_len = 247
    # print(max_len)
    data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=max_len, padding="post", value=0)
    label = tf.keras.preprocessing.sequence.pad_sequences(label, maxlen=max_len, padding="post", value=16)

    return data, label


def main():
    tag_dic, char_dic, word_dic, sentence_list, tags = read_file("../Data/weiboNER_Corpus.train")
    _, _, _, sentence_list_a, tags_a = read_file("../Data/weiboNER_2nd_conll.train")
    _, _, _, sentence_list_b, tags_b = read_file("../Data/weiboNER_2nd_conll.dev")
    _, _, _, sentence_list_c, tags_c = read_file("../Data/weiboNER_2nd_conll.test")

    convert_data("../Data/weiboNER_Corpus.train", tag_dic, char_dic, word_dic, sentence_list, tags)
    convert_data("../Data/weiboNER_2nd_conll.train", tag_dic, char_dic, word_dic, sentence_list_a, tags_a)
    convert_data("../Data/weiboNER_2nd_conll.dev", tag_dic, char_dic, word_dic, sentence_list_b, tags_b)
    convert_data("../Data/weiboNER_2nd_conll.test", tag_dic, char_dic, word_dic, sentence_list_c, tags_c)


if __name__ == "__main__":
    main()
