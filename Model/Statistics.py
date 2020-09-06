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


def get_statistics(tag_dic, label_list):
    stats = {}
    # # if label_list != None:
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

    else:
        loc = org = per = 0
        for item in label_list:
            if item[0] or tag_dic[item] == 0:
                loc += 1
            elif item[0] or tag_dic[item] == 1:
                org += 1
            elif item[0] or tag_dic[item] == 2:
                per += 1

        stats["loc"] = loc
        stats["org"] = org
        stats["per"] = per

    # elif tags != None:
    #     if len(tag_dic) == 17:
    #         gpe_nom = gpe_nam = loc_nom = loc_nam = per_nom = per_nam = org_nom = org_nam = 0
    #         for tag in tags:
    #             for item in tag:
    #                 if item.startswith("B-GPE.NAM"):
    #                     gpe_nam += 1
    #                 elif item == "B-GPE.NOM":
    #                     gpe_nom += 1
    #                 elif item == "B-LOC.NAM":
    #                     loc_nam += 1
    #                 elif item == "B-LOC.NOM":
    #                     loc_nom += 1
    #                 elif item == "B-ORG.NAM":
    #                     org_nam += 1
    #                 elif item == "B-ORG.NOM":
    #                     org_nom += 1
    #                 elif item == "B-PER.NAM":
    #                     per_nam += 1
    #                 elif item == "B-PER.NOM":
    #                     per_nom += 1
    #
    #         stats["gpe_nom"] = gpe_nom
    #         stats["gpe_nam"] = gpe_nam
    #         stats["loc_nom"] = loc_nom
    #         stats["loc_nam"] = loc_nam
    #         stats["per_nom"] = per_nom
    #         stats["per_nam"] = per_nam
    #         stats["org_nom"] = org_nom
    #         stats["org_nam"] = org_nam
    #
    #         stats["pge"] = gpe_nom + gpe_nam
    #         stats["loc"] = loc_nom + loc_nam
    #         stats["org"] = org_nom + org_nam
    #         stats["per"] = per_nom + per_nam
    #
    #     else:
    #         loc = org = per = 0
    #         for tag in tags:
    #             for item in tag:
    #                 if item == "B-LOC":
    #                     loc += 1
    #                 elif item == "B-ORG":
    #                     org += 1
    #                 elif item == "B-PER":
    #                     per += 1
    #
    #         stats["loc"] = loc
    #         stats["org"] = org
    #         stats["per"] = per
    #
    return stats


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

    precision = len(entity_ls) / len(pred_ls)
    recall = len(entity_ls) / len(label_ls)
    f1 = 2 * precision * recall / (precision + recall)
    # print(precision, "\t", recall, "\t", f1)
    return precision, recall, f1

if __name__ == "__main__":
    # tag_dic_w,_, _, tags_weibo = read_file("../Data/Chinese_Weibo_NER_Corpus.train", "weibo")
    # _, _, _, tags_weibo_origin = read_file("../Data/weiboNER_2nd_conll.train", "weibo_origin")
    # _, _, _, tags_weibo_dev = read_file("../Data/weiboNER_2nd_conll.dev", "weibo")
    # _, _, _, tags_weibo_test = read_file("../Data/weiboNER_2nd_conll.test", "weibo")
    # tag_dic_m,_, _, tags_msra_train = read_file("../Data/Chinese_MSRA_NER_Corpus.train", "msra")
    # _,_, _, tags_msra_test = read_file("../Data/Chinese_MSRA_NER_Corpus.test", "msra")
    # tag_dic_t,_, _, tags_twitter_train = read_file("../Data/English_Twitter_NER_Corpus.train", "twitter")
    # _,_, _, tags_twitter_test = read_file("../Data/English_Twitter_NER_Corpus.test", "weibo")

    for root, dirs, files in os.walk("../Data"):
        for file in files:
            file_name = os.path.join(root, file)
            # print(file_name)
            if file_name == "vec.txt":
                pass
            else:
                if "weibo" in file_name.lower():

                    tag_dic, _, _, tags = read_file(file_name, "weibo")
                    # print(tags)
                    tag_list = []
                    for tag in tags:
                        tag_list.extend(tag)
                    # print(tag_list)
                    print(get_statistics(tag_dic, tag_list))
            # break

    # get_statistics(tag_dic_w, tags=tags_weibo_origin)
    # get_statistics(tag_dic_w, tags=tags_weibo_dev)
    # get_statistics(tag_dic_w, tags=tags_weibo_test)
    # get_statistics(tag_dic_w, tags=tags_weibo)
    # get_statistics(tag_dic_m, tags=tags_msra_train)
    # get_statistics(tag_dic_m, tags=tags_msra_test)
    # get_statistics(tag_dic_t, tags=tags_twitter_train)
    # get_statistics(tag_dic_t, tags=tags_twitter_test)
    file_list = ["msra_test", "msra_train", "twitter_test", "twitter_train", "weibo_dev", "weibo_dev_origin",
                 "weibo_test", "weibo_test_origin", "weibo_train", "weibo_train_origin"]

    # for file in file_list:
    #     if file.startswith("msra"):
    #         report_perfomence(file, tag_dic_m)
    #     elif file.startswith("twitter"):
    #         report_perfomence(file, tag_dic_t)
    #     else:
    #         report_perfomence(file, tag_dic_w)


