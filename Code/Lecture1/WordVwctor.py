# from __future__ import print_function
import numpy
import re
import torch
import random
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import math


class Timer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def get_lr(filename: str):
    with open(filename, 'r') as f:
        return int(f.readline())


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    # lr *= 1 / (epoch**0.5)
    DELACY = 0.005
    epoch = epoch * DELACY + 1.0
    lr *= 2 / (1.0 + math.e**(epoch)) + (1 - 2 /
                                         (1 + math.e**(epoch))) / (epoch**0.5)
    if not math.isnan(lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


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
        # cent = torch.tanh(self.wordvec(center))
        # neg = torch.tanh(self.wordvec(negcase))
        # cont = torch.tanh(self.contextvec(context))
        cont = torch.mean(cont, dim=1, keepdim=True)
        # print("size of cent:", cent.size())
        # print("size of cont:", cont.size())
        # print("size of neg:", neg.size())
        losspos = (torch.bmm(cont, cent.transpose(1, 2)))
        lossneg = (torch.bmm(cont, neg.transpose(1, 2)))
        losspos = -torch.log(1 / (1 + torch.exp(-losspos)))
        lossneg = -torch.log(1 / (1 + torch.exp(lossneg)))
        lossneg = torch.mean(lossneg, dim=2, keepdim=True)
        # print("size of losspos: ", losspos.size())
        # print("size of lossneg: ", lossneg.size())
        loss = (losspos.mean() + lossneg.mean())
        pass
        return (loss)


DEBUG = False
# DEBUG = True

OUTPUT_INTERVAL = 50  # output every OUTPUT_INTERVAL batch
OUTPUT_TITLE_INTERVAL = 10  # print table title every OUTPUT_TITLE_INTERVAL output
SAVE_INTERVAL_BY_EPOCH = 50  # save model every SAVE_INTERVAL_BY_EPOCH epoch
FROM_EPOCH = 0
CONTEXT_WINDOW = 20  # context window to use
VECDIM = 300  # WordVec dim to use
MAX_NORM = 10
EPOCH_COUNT = 100  # how many epoch to train before rerand neg case
TRAIN_ROUND = 1000
BATCH_SIZE = 1024 * 16  # how many cases there are in each batch
NEG_COUNT = 10  # how many neg case in each case
NEG_GROUP = 30  # how many group of neg case there are for every postive case
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FREQ_POW = 0.75  # magic number
FREQ_MUL = 1000000
LR_FILE = "lr.txt"
LR = get_lr(LR_FILE)
MOMENTUM = 0.75
DAMPENING = 0.1

if DEBUG:
    DEVICE = "cpu"
    NEG_GROUP = 1  # how many group of neg case there are for every postive case
    VECDIM = 30  # WordVec dim to use

# text = """We are about to study the idea of a computational process.
# Computational processes are abstract beings that inhabit computers.
# As they evolve, processes manipulate other abstract things called data.
# The evolution of a process is directed by a pattern of rules
# called a program. People create programs to direct processes. In effect,
# we conjure the spirits of the computer with our spells. """.split()
text = []
with open("./1260.txt", 'r') as f:
    text.extend(f.read().splitlines())

text = [x.replace('*', '') for x in text]
text = [re.sub('[^ A-Za-z0-9]', '', x) for x in text]
text = [x for x in text if x != '']
raw_text = []
for x in text:
    raw_text.extend(x.split(' '))
raw_text = [str.lower(x) for x in raw_text if x != '']
text = raw_text
print(len(text))
# print(text[:200])
# quit()
wordset = [word for word in set(text)]
wordset.sort()
wordsetsize = len(wordset)
textlen = len(text)
wordtoid = {word: i for i, word in enumerate(wordset)}
with open('temp.txt', 'w') as f:
    f.write(str(wordtoid))

idtext = [wordtoid[word] for word in text]
wordcount = collections.Counter(idtext)
# print(wordcount)
modifiedwordcount = {
    wordid: int(((count / textlen)**FREQ_POW) * FREQ_MUL)
    for wordid, count in wordcount.items()
}
wordBucket = []
for wordid, count in modifiedwordcount.items():
    wordBucket.extend([wordid for i in range(count)])


def text_to_vec(text, map):
    ids = [map[x] for x in text]
    return torch.LongTensor(ids)


def rand_neg(neg_count: int):
    return torch.LongTensor(random.sample(wordBucket, neg_count))


# print(data[:5][0])
# loss_function = nn.NLLLoss()
# print(net(data[0][0], data[0][1], data[0][2]))

# for word in text:
# print(word+":"+str(wordtoid[word]))
rawdata = []
for i in range(CONTEXT_WINDOW, len(text) - CONTEXT_WINDOW):
    context = torch.LongTensor([
        idtext[j] for j in range(i - CONTEXT_WINDOW, i + CONTEXT_WINDOW + 1)
        if j != i
    ])
    target = torch.LongTensor([idtext[i]])
    rawdata.append((context, target))

net = WordVec(wordsetsize, VECDIM, CONTEXT_WINDOW, MAX_NORM)
optimizer = optim.SGD(net.parameters(),
                      lr=LR,
                      momentum=MOMENTUM,
                      dampening=DAMPENING)
net.to(DEVICE)
outputtitlecount = 0
outputcount = 0
epochcount = 0
accumateloss = 0.0
epochaccloss = 0.0
batchcnt = 0
epochloss = 2.0
curepochloss = 2.0
imporve_rate_by_epoch = 0.0
timeacc = 0.0
losses_for_shrink = []
lr_tim = 1.
shrink_cnt = 0
back_count = 0
SHRINK_RATE = 0.75
SHRINK_WINDOW = 10
SHRINK_THRESHOLD = 0.35
# optimizer.to(DEVICE)
for roundid in range(TRAIN_ROUND):
    print("resampling for round %d" % roundid)
    with Timer() as timer:
        data = []
        for case in rawdata:
            for i in range(NEG_GROUP):
                neg = torch.LongTensor(rand_neg(NEG_COUNT))
                data.append((case[0], case[1], neg))
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=BATCH_SIZE,
            #  shuffle=True)
            shuffle=False)
    print("resampled for round %d in %f sec" % (roundid, timer.elapsed))

    for epoch in range(EPOCH_COUNT):
        for i, data in enumerate(dataloader):
            with Timer() as timer:
                net.zero_grad()
                loss = net(data[0].to(DEVICE), data[1].to(DEVICE),
                           data[2].to(DEVICE))
                loss.backward()
                accumateloss += loss.item()
                epochaccloss += loss.item()
                optimizer.step()
            timeacc += timer.elapsed
            # quit()
            if outputcount % OUTPUT_INTERVAL == 0 and outputcount != 0:
                if outputtitlecount % OUTPUT_TITLE_INTERVAL == 0:
                    print(
                        "%5s %5s %5s %10s %10s %10s %15s  %15s  %10s %10s %5s %5s %10s"
                        % ("round", "epoch", "batch", "loss", "epochloss",
                           "last_eloss", "imporve_rate", "last_im_rate", "lr",
                           "momentum", "scnt", "bcnt", "sec/batch"))
                curepochloss = epochaccloss / (batchcnt + 1)
                imporve_rate = (epochloss - curepochloss) / epochloss
                print(
                    "%5d %5d %5d %10f %10f %10f %15f%% %15f%% %10f %10f %5d %5d %10f"
                    % (roundid, epoch, i, accumateloss / OUTPUT_INTERVAL,
                       curepochloss, epochloss, imporve_rate * 100.0,
                       imporve_rate_by_epoch * 100.0,
                       optimizer.param_groups[0]['lr'],
                       optimizer.param_groups[0]['momentum'], shrink_cnt,
                       back_count, timeacc / OUTPUT_INTERVAL))
                accumateloss = 0.0
                timeacc = 0.0
                outputtitlecount += 1
            outputcount += 1
            batchcnt += 1
        epochcount += 1
        if epochcount % SAVE_INTERVAL_BY_EPOCH == 0:
            torch.save(net.cpu(),
                       "data/nets/A-%d-%d.tr" % (roundid + 1, epoch + 1))
            net.to(DEVICE)
        # if (epochloss - curepochloss)/epochloss < 0.0001
        curepochloss = epochaccloss / batchcnt
        imporve_rate_by_epoch = (epochloss - curepochloss) / epochloss
        epochloss = curepochloss
        losses_for_shrink.append(imporve_rate_by_epoch)
        epochaccloss = 0.0
        batchcnt = 0
        # print(losses_for_shrink[-SHRINK_WINDOW:])
        back_count = [
            1 if l > 0 else 0 for l in losses_for_shrink[-SHRINK_WINDOW:]
        ].count(0)
        # print(back_count)
        if back_count > SHRINK_THRESHOLD * SHRINK_WINDOW:
            lr_tim *= SHRINK_RATE
            losses_for_shrink = []
            shrink_cnt += 1
        LR = get_lr(LR_FILE)
        adjust_learning_rate(optimizer, epochcount, LR * lr_tim)

torch.save(net.cpu(), "data/nets/A-Finished.tr" % (roundid, epoch))
