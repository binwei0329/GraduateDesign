#! usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

def argmax(vec):
    """
    This method finds the index of the largest value of an one-dimension vector.
    :param vec:
    :return:
    """

    _, index = torch.max(vec, 1)
    return index.item()


def log_sum_exp(vec):
    """
    This methods calculates the vec's log(sum(exp(xi)))=a+log(sum(exp(xi-a)))
    :param vec:
    :return:
    """
    max_score = vec[0, argmax(vec)]
    max_score_braodcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_braodcast)))


