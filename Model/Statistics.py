#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Graduate Project: Named Entity Recognition Task in Chinese Media
Author: Bin Wei
Description: This is part of my graduate project, which tries to explore some
            models targeting NER tasks in Chinese social media.
"""
import pickle
from Model.Preprocess import read_file

def calculate_metrics(predictions, labels, tag_dic):
    """
    This method gives the relevant evaluations on the performance of the model.
    :param predictions: predictions
    :param labels: gold labels
    :return: all the relevant performance evaluations
    """
    # Prepare three lists to store predicted labels, gold labels and correctly-predicted labels.
    pred_list = []
    label_list = []
    entity_list = []

    for i in range(len(predictions)):
        pred = predictions[i]
        label = labels[i]

        temp_pred, temp_label = [], []

        # Find the all the gold labels.
        for k in range(len(label)):
            # A label is not part of the named entity if it is tag_dic["O"].
            if label[k] != tag_dic["O"]:

                # Add such a label into temp list.
                temp_label.append(label[k])
                # If we reach the end of a sequence of labels, add the recognized item to
                # the list.
                if k == len(label) - 1:
                    label_list.append(temp_label)

            # If a label is 16, we can try to add already recognized item into the gold label list.
            else:
                # If the temp label has the length of 0, which means it has not recognized anything,
                # we continue then.
                if len(temp_label) == 0:
                    continue
                else:
                    # Add the recognized label into the gold label list and reset the temp label.
                    label_list.append(temp_label)
                    temp_label = []

        # The way how we find the predicted labels is analogous to the previous one.
        for m in range(len(pred)):
            if pred[m] != tag_dic["O"]:
                temp_pred.append(pred[m])
                if m == len(pred) - 1:
                    pred_list.append(temp_pred)
            else:
                if len(temp_pred) == 0:
                    continue
                else:
                    pred_list.append(temp_pred)
                    temp_pred = []

        # We iterate both predicted labels and gold labels, and those same items we find in
        # both lists at the same positions are the correctly predicted items.
        temp_entity = []
        for s in range(len(label)):
            if label[s] != tag_dic["O"] and pred[s] != tag_dic["O"]:

                if label[s] == pred[s]:
                    temp_entity.append(label[s])
                    if s == len(label) - 1:
                        entity_list.append(temp_entity)

            elif label[s] == pred[s] == tag_dic["O"]:

                if len(temp_entity) == 0:
                    continue
                else:
                    entity_list.append(temp_entity)
                    temp_entity = []

    # Give the relevant metrics.
    precision = len(entity_list) / len(pred_list)
    recall = len(entity_list) / len(label_list)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, entity_list, pred_list, label_list


def report_perfomence(arg, tag_dic):
    """
    This method gives relevant evaluations on the performance.
    :param arg: dataset to choose
    :param tag_dic: tag dictionary
    """
    filename = "../PickleFiles/Labels_Bilstm_crf_" + arg + ".pkl"

    with open(filename, "rb") as file:
        prediction = pickle.load(file)
        gold = pickle.load(file)

    precision, recall, f1, entity_ls, pred_ls, label_ls = calculate_metrics(prediction, gold, tag_dic)
    tag_dic_inv = {v:k for k, v in tag_dic.items()}
    # print(tag_dic_inv)
    # print(entity_ls)
    print(tag_dic)
    if len(tag_dic) == 17:
        for item in pred_ls:
            if item[0] < 8:
                            

    # print(pred_ls)
    # print(label_ls)


def get_statistics():
    tag_dic_w,_, _, tags_weibo = read_file("../Data/Chinese_Weibo_NER_Corpus.train", "weibo")
    _, _, _, tags_weibo_origin = read_file("../Data/weiboNER_2nd_conll.train", "weibo_origin")
    _, _, _, tags_weibo_dev = read_file("../Data/weiboNER_2nd_conll.dev", "weibo")
    _, _, _, tags_weibo_test = read_file("../Data/weiboNER_2nd_conll.test", "weibo")
    # print(len(tags_weibo))
    # print(len(tags_weibo_origin))
    # print(len(tags_weibo_dev))
    # print(len(tags_weibo_test))

    tag_dic_m,_, _, tags_msra_train = read_file("../Data/Chinese_MSRA_NER_Corpus.train", "msra")
    _,_, _, tags_msra_test = read_file("../Data/Chinese_MSRA_NER_Corpus.test", "msra")
    tag_dic_t,_, _, tags_twitter_train = read_file("../Data/English_Twitter_NER_Corpus.train", "twitter")
    _,_, _, tags_twitter_test = read_file("../Data/English_Twitter_NER_Corpus.test", "weibo")
    # print(len(tags_msra_test))
    # print(len(tags_msra_train))
    # print(len(tags_twitter_test))
    # print(len(tags_twitter_train))
    report_perfomence("weibo_train_origin", tag_dic_w)
    # report_perfomence("weibo_test_origin", tag_dic_w)
    # report_perfomence("weibo_dev_origin", tag_dic_w)
    # report_perfomence("weibo_train", tag_dic_w)
    # report_perfomence("weibo_test", tag_dic_w)
    # report_perfomence("weibo_dev", tag_dic_w)
    # report_perfomence("msra_train", tag_dic_m)
    # report_perfomence("msra_test", tag_dic_m)
    # report_perfomence("twitter_train", tag_dic_t)
    # report_perfomence("twitter_test", tag_dic_t)

    # with open("Statistics.txt", "w") as file:


if __name__ == "__main__":
    get_statistics()

