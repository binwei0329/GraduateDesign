#! usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from Model.Preprocess import read_file, load_char_embeddings

tag_dic, char, word_dic = read_file("../Data/Weibo_NER_Corpus.train")
char_embed_dict = load_char_embeddings("../Data/vec.txt")
