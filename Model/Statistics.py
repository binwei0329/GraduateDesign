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

def extract_labels_helper(label_list, tag_dic):
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

    label_list = extract_labels_helper(label_list, tag_dic)
    pred_list = extract_labels_helper(pred_list, tag_dic)
    entity_list = extract_labels_helper(entity_list, tag_dic)
    return entity_list, pred_list, label_list


file_list = ["msra_test", "msra_train", "twitter_test", "twitter_train", "weibo_dev", "web_dev_origin",
             "weibo_test", "weibo_test_origin", "weibo_train", "weibo_train_origin"]


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

    entity_ls, pred_ls, label_ls = calculate_metrics(prediction, gold, tag_dic)

    #     for item in pred_ls:
    #         if item[0] < 8:
    # print(stats)
    # precision = len(entity_list) / len(pred_list)
    # recall = len(entity_list) / len(label_list)
    # f1 = 2 * precision * recall / (precision + recall)

def get_statistics(tag_dic, label_list=None, tags=None):
    stats = {}
    if label_list != None:
        if len(tag_dic) == 17:
            gpe_nom = gpe_nam = loc_nom = loc_nam = per_nom = per_nam = org_nom = org_nam = 0
            for item in label_list:
                if item[0] == 0:
                    gpe_nam += 1
                elif item[0] == 1:
                    gpe_nom += 1
                elif item[0] == 2:
                    loc_nam += 1
                elif item[0] == 3:
                    loc_nom += 1
                elif item[0] == 4:
                    org_nam += 1
                elif item[0] == 5:
                    org_nom += 1
                elif item[0] == 6:
                    per_nam += 1
                elif item[0] == 7:
                    per_nom += 1

            stats["gpe_nom"] = gpe_nom
            stats["gpe_nam"] = gpe_nam
            stats["loc_nom"] = loc_nom
            stats["loc_nam"] = loc_nam
            stats["per_nom"] = per_nom
            stats["per_nam"] = per_nam
            stats["org_nom"] = org_nom
            stats["org_nam"] = org_nam

        else:
            loc = org = per = 0
            for item in label_list:
                if item[0] == 0:
                    loc += 1
                elif item[0] == 1:
                    org += 1
                elif item[0] == 2:
                    per += 1

            stats["loc"] = loc
            stats["org"] = org
            stats["per"] = per
    # {'B-LOC': 0, 'B-ORG': 1, 'B-PER': 2, 'I-LOC': 3, 'I-ORG': 4, 'I-PER': 5, 'O': 6}

    elif tags != None:
        if len(tag_dic) == 17:
            gpe_nom = gpe_nam = loc_nom = loc_nam = per_nom = per_nam = org_nom = org_nam = 0
            for tag in tags:
                for item in tag:
                    if item == "B-GPE.NAN":
                        gpe_nam += 1
                    elif item == "B-GPE.NOM":
                        gpe_nom += 1
                    elif item == "B-LOC.NAM":
                        loc_nam += 1
                    elif item == "B-LOC.NOM":
                        loc_nom += 1
                    elif item == "B-ORG.NAM":
                        org_nam += 1
                    elif item == "B-ORG.NOM":
                        org_nom += 1
                    elif item == "B-PER.NAM":
                        per_nam += 1
                    elif item == "B-PER.NOM":
                        per_nom += 1

            stats["gpe_nom"] = gpe_nom
            stats["gpe_nam"] = gpe_nam
            stats["loc_nom"] = loc_nom
            stats["loc_nam"] = loc_nam
            stats["per_nom"] = per_nom
            stats["per_nam"] = per_nam
            stats["org_nom"] = org_nom
            stats["org_nam"] = org_nam

        else:
            loc = org = per = 0
            for tag in tags:
                for item in tag:
                    if item == "B-LOC":
                        loc += 1
                    elif item == "B-ORG":
                        org += 1
                    elif item == "B-PER":
                        per += 1

            stats["loc"] = loc
            stats["org"] = org
            stats["per"] = per

    print(stats)


if __name__ == "__main__":
    tag_dic_w,_, _, tags_weibo = read_file("../Data/Chinese_Weibo_NER_Corpus.train", "weibo")
    _, _, _, tags_weibo_origin = read_file("../Data/weiboNER_2nd_conll.train", "weibo_origin")
    _, _, _, tags_weibo_dev = read_file("../Data/weiboNER_2nd_conll.dev", "weibo")
    _, _, _, tags_weibo_test = read_file("../Data/weiboNER_2nd_conll.test", "weibo")
    tag_dic_m,_, _, tags_msra_train = read_file("../Data/Chinese_MSRA_NER_Corpus.train", "msra")
    _,_, _, tags_msra_test = read_file("../Data/Chinese_MSRA_NER_Corpus.test", "msra")
    tag_dic_t,_, _, tags_twitter_train = read_file("../Data/English_Twitter_NER_Corpus.train", "twitter")
    _,_, _, tags_twitter_test = read_file("../Data/English_Twitter_NER_Corpus.test", "weibo")
    # print(tags_weibo)
    #
    # print(tag_dic_w)
    # print(tag_dic_m)
    # print(tag_dic_t)
    #
    # tag_dic_inv = {v: k for k, v in tag_dic_w.items()}
    # print(tag_dic_inv)
    #
    # m = {v: k for k, v in tag_dic_m.items()}
    # print(m)
    #
    # t =  {v: k for k, v in tag_dic_t.items()}
    # print(t)

    get_statistics(tag_dic_w, tags=tags_weibo_origin)

