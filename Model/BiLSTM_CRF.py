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


class BiLSTM_CRF(nn.module):
    def __init__(self, vocab_size, tag_dict, embed_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.tag_dict = tag_dict
        self.tag_size = len(tag_dict)

        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden_to_tag = nn.Linear(hidden_dim, self.tag_size)

        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.transitions.data[tag_dict[STOP_TAG],:] = -10000
        self.transitions.data[:, tag_dict[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden // 2))


    def _forward_pass(self):



