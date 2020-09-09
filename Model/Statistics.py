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
from Model.Preprocess import read_file

def extract_labels_helper(label_list, tag_dic):
    """
    This method extracts labels from predictions of the model, and
    this method is aimed for situations like two or more predicted tags neighboring
    each other, like "B-ORG I-ORG B-PER I-PER".
    :param label_list: predicted labels
    :param tag_dic: tag dictionary
    :return: extracted labels
    """
    label_ls = []
    num = (len(tag_dic) - 1) // 2
    for label in label_list:
        index = []
        for l in range(len(label)):
            if label[l] < num:
                index.append(l)
        if len(index) == 1:
            label_ls.append(label)
        elif len(index) == 2:
            label_ls.append(label[:index[1]])
            label_ls.append(label[index[1]:])
        elif len(index) == 3:
            label_ls.append(label[:index[1]])
            label_ls.append(label[index[1]:index[2]])
            label_ls.append(label[index[2]:])
        elif len(index) == 4:
            label_ls.append(label[:index[1]])
            label_ls.append(label[index[1]:index[2]])
            label_ls.append(label[index[2]:index[3]])
            label_ls.append(label[index[3]:])

    return label_ls


def extract_labels(predictions, labels, tag_dic):
    """
    This method entities from both labels and predictions.
    :param predictions: predictions
    :param labels: gold labels
    :tag_dic: tag dictionary for reference
    :return: extracted entities
    """
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

    label_list = extract_labels_helper(label_list, tag_dic)
    pred_list = extract_labels_helper(pred_list, tag_dic)
    entity_list = extract_labels_helper(entity_list, tag_dic)

    return entity_list, pred_list, label_list


def get_statistics(tag_dic, label_list):
    """
    This method gives the stats of entity of each type.
    :param tag_dic: tag dictionary for reference
    :param label_list: tag list or label list
    :return: statistical info of every type of tag
    """
    stats = {}
    if len(tag_dic) == 17:
        gpe_nom = gpe_nam = loc_nom = loc_nam = per_nom = per_nam = org_nom = org_nam = 0
        for item in label_list:
            if item[0] == 0 or tag_dic[item] == 0:
                gpe_nam += 1
            elif item[0] == 1 or tag_dic[item] == 1:
                gpe_nom += 1
            elif item[0] == 2 or tag_dic[item] == 2:
                loc_nam += 1
            elif item[0] == 3 or tag_dic[item] == 3:
                loc_nom += 1
            elif item[0] == 4 or tag_dic[item] == 4:
                org_nam += 1
            elif item[0] == 5 or tag_dic[item] == 5:
                org_nom += 1
            elif item[0] == 6 or tag_dic[item] == 6:
                per_nam += 1
            elif item[0] == 7 or tag_dic[item] == 7:
                per_nom += 1

        stats["gpe_nom"] = gpe_nom
        stats["gpe_nam"] = gpe_nam
        stats["loc_nom"] = loc_nom
        stats["loc_nam"] = loc_nam
        stats["per_nom"] = per_nom
        stats["per_nam"] = per_nam
        stats["org_nom"] = org_nom
        stats["org_nam"] = org_nam
        stats["gpe"] = gpe_nam + gpe_nom
        stats["loc"] = loc_nam + loc_nom
        stats["org"] = loc_nam + loc_nom
        stats["per"] = per_nam + per_nom

    else:
        loc = org = per = 0
        for item in label_list:
            if item[0] == 0 or tag_dic[item] == 0:
                loc += 1
            elif item[0] == 1 or tag_dic[item] == 1:
                org += 1
            elif item[0] == 2 or tag_dic[item] == 2:
                per += 1

        stats["loc"] = loc
        stats["org"] = org
        stats["per"] = per

    return stats


def report_perfomence(arg, tag_dic, cond=None):
    """
    This method gives relevant evaluations on the performance.
    :param arg: dataset to choose
    :param tag_dic: tag dictionary
    :return: precision, recall, f1
    """
    if cond == None:
        filename = "../PickleFiles/Labels_Bilstm_crf_" + arg + ".pkl"
    else:
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
    keyword = sorted(["weibo", "msra", "twitter"])
    file_list = ["msra_test", "msra_train", "twitter_test", "twitter_train", "weibo_dev", "weibo_dev_origin",
                 "weibo_test", "weibo_test_origin", "weibo_train", "weibo_train_origin"]

    # Write relevant stats into the file.
    if not os.path.exists("../Data/Stats.txt"):
        tag_dic_store = {}
        with open("../Data/Stats.txt", "w") as f:
            for root, dirs, files in os.walk("../Data"):
                for file in files:
                    file_name = os.path.join(root, file)
                    if file_name == "vec.txt":
                        pass
                    else:
                        tag_dic = None
                        for kw in keyword:
                            if kw in file_name.lower():
                                tag_dic, _, _, tags = read_file(file_name, kw)
                                tag_dic_store[kw] = tag_dic
                                tag_list = []
                                for tag in tags:
                                    tag_list.extend(tag)
                                stats = get_statistics(tag_dic, tag_list)
                                f.write(file_name)
                                f.write("\n")
                                f.write(str(stats))
                                f.write("\n\n")

        # Write tag dictionaries for later reference.
        with open("../Data/Tag_Dictionary.txt", "w") as f:
            f.write(str(tag_dic_store))

    if os.path.exists("../Data/Tag_Dictionary.txt"):
        with open("../Data/Tag_Dictionary.txt", "r") as f:
            tag_dic_store = eval(f.readline())
            tag_dic_m = tag_dic_store["msra"]
            tag_dic_t = tag_dic_store["twitter"]
            tag_dic_w = tag_dic_store["weibo"]

    # Add performance of model on every dataset to the stats file.
    if os.path.exists("../Data/Stats.txt"):
        with open("../Data/Stats.txt", "a") as f:
            f.write("BiLSTM_CRF Results:\n")
            f.write("Precision\t\tRecall\t\tF1\t\t\n")
            for file in file_list:
                if file.startswith("msra"):
                    f.write("MSRA Corpus Performance:\n")
                    precision, recall, f1 = report_perfomence(file, tag_dic_m)
                    f.write(file)
                    f.write("\n")
                    f.write(precision)
                    f.write("\t\t")
                    f.write(recall)
                    f.write("\t\t")
                    f.write(f1)
                    f.write("\t\t\n")

                elif file.startswith("twitter"):
                    f.write("Twitter Corpus Performance:\n")
                    precision, recall, f1 = report_perfomence(file, tag_dic_t)
                    f.write(file)
                    f.write("\n")
                    f.write(precision)
                    f.write("\t\t")
                    f.write(recall)
                    f.write("\t\t")
                    f.write(f1)
                    f.write("\t\t\n")

                else:
                    f.write("Weibo Corpus Performance:\n")
                    precision, recall, f1 = report_perfomence(file, tag_dic_w)
                    f.write(file)
                    f.write("\n")
                    f.write(precision)
                    f.write("\t\t")
                    f.write(recall)
                    f.write("\t\t")
                    f.write(f1)
                    f.write("\t\t\n")

            f.write("IDCNN_CRF Results:\n")
            f.write("Precision\t\tRecall\t\tF1\t\t\n")
            for file in file_list:
                if file.startswith("weibo"):
                    f.write("Weibo Corpus Performance:\n")
                    precision, recall, f1 = report_perfomence(file, tag_dic_w, "idcnn_crf")
                    f.write(file)
                    f.write("\n")
                    f.write(precision)
                    f.write("\t\t")
                    f.write(recall)
                    f.write("\t\t")
                    f.write(f1)
                    f.write("\t\t\n")

    print("All done.")

