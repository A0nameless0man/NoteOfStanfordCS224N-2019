import torch
import torch.nn as nn
# import random
from torch import LongTensor
from random import sample
from multiprocessing import Pool, cpu_count
import functools


class WordVec(nn.Module):
    def __init__(self, size: int, length: int, window: int, MAX_NORM: float):
        super(WordVec, self).__init__()
        self.vecsize = size
        self.dirlength = length
        self.window = window
        self.wordvec = nn.Embedding(size, length, max_norm=MAX_NORM)
        # self.wordvec.weight.data.uniform_(-0.5 / length, 0.5 / length)
        self.contextvec = nn.Embedding(size, length, max_norm=MAX_NORM)
        # self.contextvec.weight.data.uniform_(-0.5 / length, 0.5 / length)

    def forward(self, context, center, negcase):
        cent = self.wordvec(center)
        neg = self.wordvec(negcase)
        cont = self.contextvec(context)
        cont = torch.mean(cont, dim=1, keepdim=True)
        losspos = (torch.bmm(cont, cent.transpose(1, 2)))
        lossneg = (torch.bmm(cont, neg.transpose(1, 2)))
        losspos = -torch.log(1 / (1 + torch.exp(-losspos)))
        lossneg = -torch.log(1 / (1 + torch.exp(lossneg)))
        lossneg = torch.mean(lossneg, dim=2, keepdim=True)
        loss = (losspos.mean() + lossneg.mean())
        pass
        return (loss)


def neg_sample(NEG_COUNT, wordBucket, case):
    return (case[0], case[1], LongTensor(sample(wordBucket, NEG_COUNT)))


def batch_neg_sample(NEG_COUNT, wordBucket, cases, process_cnt=cpu_count()):
    with Pool(process_cnt) as p:
        return p.map(functools.partial(neg_sample, NEG_COUNT, wordBucket),
                     cases)
    # return map(functools.partial(neg_sample, NEG_COUNT, wordBucket), cases)


def build_postive_samples(CONTEXT_WINDOW, raw):
    samples = []
    append = samples.append
    for i in range(CONTEXT_WINDOW, len(raw) - CONTEXT_WINDOW):
        context = LongTensor([
            raw[j] for j in range(i - CONTEXT_WINDOW, i + CONTEXT_WINDOW + 1)
            if j != i
        ])
        target = LongTensor([raw[i]])
        append((context, target))
    return samples


def build_word_bucket(word_freq, FREQ_POW: float, FREQ_MUL: int):
    return [
        word for word, freq in word_freq.items()
        for _ in range(int((freq)**FREQ_POW * FREQ_MUL))
    ]
