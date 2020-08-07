# # # -*- coding: utf-8 -*-
# # import torch
# # import torch.autograd as autograd
# # import torch.nn as nn
# # import torch.optim as optim
# # from Model.Preprocess import read_file, load_char_embeddings
# #
# # torch.manual_seed(1)
# # # ========================= 相关函数 ==========================
# # def argmax(vec):
# #     # 得到最大值的标签
# #     # vec: [1, tagset_size]
# #     _, idx = torch.max(vec, 1)
# #     return idx.item()
# #
# # def prepare_sequence(seq, to_ix):
# #     # 将一句话中的字转换为索引
# #     # seq: [vocab_size]
# #     # to_ix: dict
# #     idxs = [to_ix[word] for word in seq]
# #     return torch.tensor(idxs, dtype=torch.long)
# #
# # def log_sum_exp(vec):
# #     # 计算$\log\sum{e^{vec_i}}$
# #     # vec: [1, tagset_size]
# #     # 为防止指数运算溢出,先将vec中元素减去最大值,最后在结果中加上最大值
# #     max_score = vec[0, argmax(vec)]
# #     # max_score的维度可以不扩展,直接利用广播机制也行 Pytorch.view用法:https://www.jianshu.com/p/67afe38297f0,基本上就是numpy中的resize。
# #     max_score_broadcast = max_score.view([1, -1]).expand([1, vec.size()[1]])
# #     # 最后return返回的是一个一维的tensor
# #     return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
# #
# # # ====================== BILSTM实现 ==================================
# # class BiLSTM_CRF(nn.Module):
# #     def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
# #         super(BiLSTM_CRF, self).__init__()
# #         self.vocab_size = vocab_size
# #         self.tag_to_ix = tag_to_ix   # 包括了表示开始开始和结束的标签
# #         self.embedding_dim = embedding_dim
# #         self.hidden_dim = hidden_dim
# #         self.tagset_size = len(tag_to_ix)
# #
# #         self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
# #         self.BiLSTM = nn.LSTM(embedding_dim, hidden_size=hidden_dim//2,
# #                               bidirectional=True, num_layers=1)
# #         # hidden_size除以2是为了使BiLSTM的输出维度依然是hidden_size,而不用乘以2
# #
# #         # 通过将BiLSTM的输出接上nn.Linear得到发射分数hidden2tag: [seq, 1, tagset_size]
# #         # batch_size在整个程序中的维度是1
# #         self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
# #
# #         # CRF层学习的就是一个转移分数transitions: [tagset_size, tagset_size]
# #         # transitions[i]表示的是**从j=1,2,...,tagset_size**转移到第i个标签的分数
# #         # 而不能理解为**从第i个标签注意到j=1,2,...,tagset_size**的分数
# #         self.transitions = nn.Parameter(torch.randn([self.tagset_size, self.tagset_size]))
# #
# #         # 用表示开始和结束的特殊字符找到需要识别的句子的开始和结束,用一个负无穷的数约束,这样exp
# #         # 一个负无穷的数,其得分就是0
# #         self.transitions.data[tag_to_ix[START_TAG], :] = -10000
# #         self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
# #
# #         # 初始化BiLSTM的隐层单元,可以不在这里初始化,因为forward函数内又写了一句
# #         self.hidden = self.init_hidden()
# #
# #     def init_hidden(self):
# #         # 隐层单元的维度为:
# #         # [num_layers * num_directions, batch, hidden_size]
# #         return (torch.randn([2, 1, self.hidden_dim//2]),
# #                 torch.randn([2, 1, self.hidden_dim//2]))
# #
# #     # 得到发射分数
# #     def _get_lstm_features(self, sentence):
# #         self.hidden = self.init_hidden()
# #         embeds = self.word_embeds(sentence).view([len(sentence), 1, -1])
# #         BiLSTM_out, self.hidden = self.BiLSTM(embeds, self.hidden)
# #         # BiLSTM_out的输出本来是[seq_len, batch_size, hidden_dim]
# #         # 由于batch_size为1,view成了二维的
# #         BiLSTM_out = BiLSTM_out.view([len(sentence), self.hidden_dim])
# #         BiLSTM_feats = self.hidden2tag(BiLSTM_out)
# #         return BiLSTM_feats
# #
# #     # 得到正确路径的分数,即公式中的S(X, y)
# #     # 只需要:
# #     # (1)依次得到标签tags对应的转移分数
# #     # (2)加上feats对应的发射分数就行了
# #     def _score_sentence(self, feats, tags):
# #         # feats: [seq_len, tagset_size]
# #         # tags:  [tagset_size]
# #         score = torch.zeros([1])
# #         # 由于tags里不含表示开始和结束的特殊字符,而转移分数的矩阵内是有的
# #         # 因此首先在tags添加了表示开始的特殊字符
# #         tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]],
# #                                        dtype=torch.long), tags])
# #         for i, feat in enumerate(feats):
# #             # transitions[tags[i+1], tags[i]]表示第i个标签转移到第i+1个标签的转移分数
# #             score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
# #         score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
# #         return score
# #
# #     # 得到所有路径的分数,即公式中的$\log\sum_{\tilde{y}\in Y_X}e^{S(X, \tilde{y})}$
# #     # 这里采用类似动态规划的做法,因为要求解出所有可能路径的分数再求和时间复杂度太高了
# #     # 可以依次从前往后计算出每一条路径上的分数,当计算下一条路径时,直接加上前一次计算得到的路径分数就行
# #     # 因此在前向计算的过程中需要保存前一次计算的路径得分,程序中用forward_var表示的,维度为:[1 tagset_size]
# #     def _forward_alg(self, feats):
# #         # feats: [seq_len, tagset_size]
# #         # 初始化forward_var, 并且开始位置的分数为0, 迫使转移矩阵学到START_TAG的得分最高
# #         forward_var = torch.full([1, self.tagset_size], -10000.)
# #         forward_var[0][self.tag_to_ix[START_TAG]] = 0.
# #
# #         # 前向过程计算分数的前向是针对seq_len而言的,每一次存储都是在每一次seq_len的结束存储的
# #         for feat in feats:
# #             # forward_var_t表示每一次前向过程中的分数
# #             # forward_var_t与forward_var不同,forward_var_t每一次前向过程中需要更新,但是
# #             # forward_var是累加的
# #             forward_var_t = []
# #             # 这个for循环计算的是在t时刻前向计算过程中,所有标签到某个具体标签的得分
# #             #   x0   |  x1    x2    x3
# #             # ------>---->----->----->------->
# #             #  START | START START START
# #             #   y1   |  y1    y1    y1
# #             #   y2   |  y2    y2    y2
# #             #   y3   |  y3    y3    y3
# #             #  STOP  | STOP  STOP  STOP
# #             # 假设feats = [[x1, x2, x3]]
# #             # 可能的标签为{START, y1, y2, y3, STOP}
# #             # 假如此时feat = x1
# #             # 则此时下面的for循环需要依次计算:
# #             # (1) {START, y1, y2, y3, STOP}到START的总分数作为forward的**第0个元素**
# #             # (2) {START, y1, y2, y3, STOP}到y1的总分数作为forward的*****第1个元素**
# #             # ............................................................
# #             # (5) {START, y1, y2, y3, STOP}到STOP的总分数作为forward的***第4个元素**
# #             # ======================== 细节: 如何采用动态规划思想 ================
# #             # 对于计算(2) **{START, y1, y2, y3, STOP}到y1的总分数作为forward的第1个元素** 时
# #             # 需要分别加上forward在前一时刻(t-1时刻)的得分,举例: 计算START到y1的分数S(START, y1):
# #             # S(START, y1) = forward[0] + E(x1, y1) + T(START, y1)
# #             # 其中forward[0]表示t-1时刻所有到达START所有路径的得分
# #             # E(x1, y1)与T(START, y1)分别表示发射分数和转移分数
# #             for next_tag in range(self.tagset_size):
# #                 # 复制emit_score的目的是因为对于t-1时刻无论何种方式到达标签next_tag,其对应的发射分数不会变
# #                 # 变的是转移分数
# #                 emit_score = feat[next_tag].view([1, -1]).expand([1, self.tagset_size])
# #                 trans_score = self.transitions[next_tag].view([1, -1])
# #                 # ===================== 这里计算的就是前面说的细节处的计算=======================
# #                 next_tag_var = forward_var + trans_score + emit_score
# #                 forward_var_t.append(log_sum_exp(next_tag_var).view([1]))
# #             forward_var = torch.cat(forward_var_t).view([1, -1])
# #         # 计算最后到达STOP_END的得分,此时只有转移分数
# #         forward_var += self.transitions[self.tag_to_ix[STOP_TAG]]
# #         forward_var = log_sum_exp(forward_var)
# #         return forward_var
# #
# #     # 计算CRF的损失函数
# #     # $ Loss = -(S(X, y) - \log\sum_{\tilde{y}\in Y_X}e^{S(X, \tilde{y})} $
# #     #        = self._forward_alg(feats) - self._score_sentence(feats, tags)
# #     def neg_log_likelihood(self, sentence, tags):
# #         feats = self._get_lstm_features(sentence)
# #         forward_score = self._forward_alg(feats)
# #         gold_score = self._score_sentence(feats, tags)
# #         return forward_score - gold_score
# #
# #     # viterbi解码时,也是运用了动态规划的思想,其实和self._forward_alg类似,
# #     def _viterbi_decode(self, feats):
# #         # 初始化forward_var,并且开始位置的分数为0,确保一定是从START_TAG开始的,
# #         # 因为 $e^{-10000}<<e^0$
# #         forward_var = torch.full([1, self.tagset_size], -10000.)
# #         forward_var[0][self.tag_to_ix[START_TAG]] = 0
# #         # backpointers用来计算每一个时刻每一个标签对应的最佳路径
# #         backpointers = []
# #         for feat in feats:
# #             backpointers_t = []  # t时刻的最佳路径
# #             forward_var_t = []  # t时刻的最佳路径的得分
# #             #   x0   |  x1    x2    x3
# #             # ------>---->----->----->------->
# #             #  START | START START START
# #             #   y1   |  y1    y1    y1
# #             #   y2   |  y2    y2    y2
# #             #   y3   |  y3    y3    y3
# #             #  STOP  | STOP  STOP  STOP
# #             # 当feat=x2时,假如在t-1时刻START到{START, y1, y2, y3, STOP}的路径得分最大
# #             # 此时,需要求t时刻达到{START, y1, y2, y3, STOP}的路径得分
# #             # 由于此时发射分数都是一样的,因此只要比较转移分数就行
# #             # 举例：计算{START, y1, y2, y3, STOP}到START的最大路径
# #             # 计算max(T(START, START)+forward_var[0], T(y1, START)+forward_var[1], T(y2, START), ...)
# #             # 假设T(y1, START)+forward_var[1]最大,此时y1对应的索引(也就是1)被记录在backpointers_t中,
# #             # 值T(y1, START)+forward_var[1]+E(x2, y1)被记录在forward_var_t中
# #             for next_tag in range(self.tagset_size):
# #                 next_tag_var = forward_var + self.transitions[next_tag]
# #                 best_tag_id = argmax(next_tag_var)
# #                 backpointers_t.append(best_tag_id)
# #                 forward_var_t.append(next_tag_var[0][best_tag_id].view([1]))
# #             # 更新forward_var
# #             forward_var = (torch.cat(forward_var_t) + feat).view([1, -1])
# #             # 添加backpointers
# #             backpointers.append(backpointers_t)
# #
# #         # 计算到STOP_TAG的最优路径,其得分也就是最优路径的得分
# #         forward_var += self.transitions[self.tag_to_ix[STOP_TAG]]
# #         best_tag_id = argmax(forward_var)
# #         path_score = forward_var[0][best_tag_id]
# #
# #         # 通过backpointers逆序找到最佳路径
# #         best_path = [best_tag_id]
# #         for backpointers_t in reversed(backpointers):
# #             best_tag_id = backpointers_t[best_tag_id]
# #             best_path.append(best_tag_id)
# #         # 弹出START_TAG
# #         start = best_path.pop()
# #         assert self.tag_to_ix[START_TAG] == start
# #         best_path.reverse()
# #         return path_score, best_path
# #
# #     def forward(self, sentence):
# #         # 得到发射分数
# #         BiLSTM_feats = self._get_lstm_features(sentence)
# #
# #         # 通过viterbi找出最佳路径
# #         score, best_path = self._viterbi_decode(BiLSTM_feats)
# #         return score, best_path
# #
# # if __name__ == '__main__':
# #     START_TAG = "<START>"
# #     STOP_TAG = "<STOP>"
# #     EMBEDDING_DIM = 5
# #     HIDDEN_DIM = 4
# #
# #     tag_dic, char, word_dic, sentence_list, tags = read_file("../Data/Weibo_NER_Corpus.train")
# #
# #     training_data = []
# #     for i in range(len(tags)):
# #         training_data.append((sentence_list[i], tags[i]))
# #
# #
# #     # training_data = [(
# #     #     "the wall street journal reported today that apple corporation made money".split(),
# #     #     "B I I I O O O B I O O".split()
# #     # ), (
# #     #     "georgia tech is a university in georgia".split(),
# #     #     "B I O O O O B".split()
# #     # )]
# #
# #     word_to_ix = {}
# #     for sentence, tags in training_data:
# #         for word in sentence:
# #             if word_to_ix.get(word) is None:
# #                 word_to_ix[word] = len(word_to_ix)
# #
# #     tag_to_ix = tag_dic
# #     tag_to_ix[START_TAG] = 17
# #     tag_to_ix[STOP_TAG] = 18
# #
# #     # tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
# #     model = BiLSTM_CRF(vocab_size=len(word_to_ix),
# #                        tag_to_ix=tag_to_ix,
# #                        embedding_dim=EMBEDDING_DIM,
# #                        hidden_dim=HIDDEN_DIM, )
# #     optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# #
# #     # 仅仅为了print
# #     with torch.no_grad():
# #         precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# #         precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
# #         print(model(precheck_sent))
# #
# #     # 开始训练
# #     for epoch in range(10):
# #         # 注意: 这里的batch_size为1
# #         for sentence, tags in training_data:
# #             model.zero_grad()
# #             sentence_in = prepare_sequence(sentence, word_to_ix)
# #             targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
# #
# #             loss = model.neg_log_likelihood(sentence_in, targets)
# #             loss.backward()
# #             optimizer.step()
# #
# #     # 预测
# #     with torch.no_grad():
# #         precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# #         print(model(precheck_sent))
# #
# #
# # # import tensorflow as tf
# # # import tensorflow_addons as tf_ad
# # # from Model.Preprocess import read_file, load_char_embeddings
# # #
# # # class BiLSTM_CRF(tf.keras.Model):
# # #     def __init__(self, vocab_size, label_size, hidden_dim, embedding_dim):
# # #         super(BiLSTM_CRF, self).__init__()
# # #         self.hidden_dim = hidden_dim
# # #         self.vocab_size = vocab_size
# # #         self.label_size = label_size
# # #
# # #         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
# # #         self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))
# # #         self.dense = tf.keras.layers.Dense(label_size)
# # #
# # #         self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))
# # #         self.dropout = tf.keras.layers.Dropout(0.5)
# # #
# # #     # @tf.function
# # #     def call(self, text, labels=None, training=None):
# # #         text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
# # #         # -1 change 0
# # #         inputs = self.embedding(text)
# # #         inputs = self.dropout(inputs, training)
# # #         logits = self.dense(self.biLSTM(inputs))
# # #
# # #         if labels is not None:
# # #             label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
# # #             log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits,
# # #                                                                                    label_sequences,
# # #                                                                                    text_lens,
# # #                                                                                    transition_params=self.transition_params)
# # #             return logits, text_lens, log_likelihood
# # #         else:
# # #             return logits, text_lens
# # #
# # #
# # #
# # # if __name__ == "__main__":
# # #         tag_dic, char, word_dic, sentence_list, tags = read_file("../Data/Weibo_NER_Corpus.train")
# # #         print(word_dic)
#
# import torch
# DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available()) # True
# print(torch.cuda.device_count()) # 1
# print(DEVICE)
#
# # a = torch.randn(10, 3, 100).cuda()
# # print(type(a))
#
# # word_embed = torch.nn.Embedding(20, 20)
# # lstm = torch.nn.LSTM(20, 10, num_layers=1, bidirectional=True)
# # print(type(lstm))
#
# score = torch.zeros(1)  # torch.Tensor
# # print(type(score))
#
# init = torch.full((1, 50), -10000.).cuda()
# init[0][0] = 0.
# # print(type(init))
# _, idx = torch.max(init, 1)
#
# print(idx)
# import tensorf


# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 7:15 下午
# @Author: wuchenglong

import os
import torch
import codecs
import pickle
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
from torch.utils import data
import torch.autograd as autograd
from Model.Preprocess import convert_data, format_data

import torch
torch.cuda.set_device(0)

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
def argmax(vec):
    '''
    计算一维vec最大值的坐标
    '''
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    '''
    计算vec的 log(sum(exp(xi)))=a+log(sum(exp(xi-a)))
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class Model(nn.Module):

    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)

        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim).cuda()
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
        #                     num_layers=1, bidirectional=True).cuda()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).cuda()
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j = trans[i][j]
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag2id[START_TAG], :] = -10000
        self.transitions.data[:, tag2id[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).cuda(),
                torch.randn(2, 1, self.hidden_dim // 2).cuda())
        # return (torch.randn(2, 1, self.hidden_dim // 2),
        #         torch.randn(2, 1, self.hidden_dim // 2))


    def _forward_alg(self, feats):
        '''
        :param feats: LSTM+hidden2tag的输出
        :return: 所有tag路径的score和
        forward_var：之前词的score和
        '''
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).cuda()
        # init_alphas = torch.full((1, self.tagset_size), -10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag2id[START_TAG]] = 0. #

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:   #every word
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size): #every word's tag
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size).cuda()
                # emit_score = feat[next_tag].view(
                #     1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).cuda()
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1).cuda())
                # alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1).cuda()
            # forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag2id[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        # alpha = log_sum_exp(terminal_var)

        return alpha


    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1).cuda()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim).cuda()
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # lstm_feats = self.hidden2tag(lstm_out).cuda()
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence 当前句子的tag路径score
        score = torch.zeros(1).cuda()
        tags = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long).cuda(), tags])
        # tags = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long), tags])

        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2id[STOP_TAG], tags[-1]]
        return score


    def _viterbi_decode(self, feats):
        backpointers = [] #路径保存

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        # init_vvars = torch.full((1, self.tagset_size), -10000.)

        init_vvars[0][self.tag2id[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats: # for every word
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size): #for every possible tag of the word
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1).cuda())
                # viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2id[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2id[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path


    def forward(self, sentence, tags):
        # feats = self._get_lstm_features(sentence).cpu()
        feats = self._get_lstm_features(sentence)

        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score


    def test(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# def determine_device():
#     """
#     This function evaluates whether a GPU is accessible at the system and
#     returns it as device to calculate on, otherwise it returns the CPU.
#     :return: The device where tensor calculations shall be made on
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#     print()
#
#     # Additional Info when using cuda
#     if device.type == "cuda":
#         print(torch.cuda.get_device_name(0))
#         print("Memory Usage:")
#         print("\tAllocated:",
#               round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
#         print("\tCached:   ", round(torch.cuda.memory_cached(0) / 1024 ** 3, 1),
#               "GB")
#         print()
#     return device

if __name__ == "__main__":
    EMBED_DIM = 100
    HIDDEN_DIM = 200
    EPOCH = 5
    LEARNING_RATE = 0.005

    # config = dict()
    # config["device"] = "GPU"
    # if config["device"] == "GPU":
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # convert_data("../Data/Weibo_NER_Corpus.train")
    # _, _, _, _, _, data, label = {}, {}, {}, [], [], [], []
    # with open ("../Data/BiLSTM_CRF_data.pkl", "rb") as file:
    #     tag_dic = pickle.load(file)
    #     char_dic = pickle.load(file)
    #     word_dic = pickle.load(file)
    #     sentence_list = pickle.load(file)
    #     tags = pickle.load(file)
    #     data = pickle.load(file)
    #     label = pickle.load(file)
    # print(data, "\n", label)
    # data, label = format_data(data, label)
    # tf.print(data)
    # tf.print(label)
    # tag_dic[START_TAG] = len(tag_dic)
    # tag_dic[STOP_TAG] = len(tag_dic)

    # print(data[:5])
    # print(label[:5])
    # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Model(len(char_dic) + 1, tag_dic, EMBED_DIM, HIDDEN_DIM).cuda()
    # # model = Model(len(char_dic) + 1, tag_dic, EMBED_DIM, HIDDEN_DIM)
    #
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    #
    # # dataset = list(zip(data, label))
    # # train_loader = data.Dataloader(dataset)
    # for e in range(EPOCH):
    #     index = 0
    #     for sentence, tag in zip(data, label):
    #         index += 1
    #         model.zero_grad()
    #
    #         sentence = torch.tensor(sentence, dtype=torch.long).cuda()
    #         #
    #         tag = torch.tensor(tag, dtype=torch.long).cuda()
    #         # sentence = torch.tensor(sentence, dtype=torch.long)
    #
    #         # tag = torch.tensor(tag, dtype=torch.long)
    #
    #         loss = model(sentence, tag)
    #         # print(type(loss))
    #         loss.backward()
    #         optimizer.step()
    #
    #         if index % 20 == 0:
    #             print("epoch " , e, " index ", index)
    #             print(loss.item())
    #
