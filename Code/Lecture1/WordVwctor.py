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
import os
import gc
from torch.multiprocessing import Pool
import functools


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
        return float(f.readline())


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


def neg_sample(NEG_COUNT, wordBucket, case):
    return (case[0], case[1],
            torch.LongTensor(random.sample(wordBucket, NEG_COUNT)))

def loadDataFile(folder, sufix):
    dirInfo = os.walk(folder)
    files = []
    for path, dir_list, file_list in dirInfo:
        for file_name in file_list:
            if file_name.endswith(sufix):
                files.append(os.path.join(path, file_name))
    return files


def loadFile(path):
    text = []
    with open(path, 'r') as f:
        text.extend(f.read().splitlines())

    text = [x.replace('*', '') for x in text]
    text = [re.sub('[^ A-Za-z0-9]', '', x) for x in text]
    text = [x for x in text if x != '']
    raw_text = []
    for x in text:
        raw_text.extend(x.split(' '))
    raw_text = [str.lower(x) for x in raw_text if x != '']
    text = raw_text
    return text


def getWordSetFromContent(content):
    wordset = [word for word in set(content)]
    return wordset


def main():

    DEBUG = False
    # DEBUG = True

    OUTPUT_INTERVAL = 25  # output every OUTPUT_INTERVAL batch
    OUTPUT_TITLE_INTERVAL = 10  # print table title every OUTPUT_TITLE_INTERVAL output
    SAVE_INTERVAL_BY_EPOCH = 50  # save model every SAVE_INTERVAL_BY_EPOCH epoch
    FROM_EPOCH = 0
    CONTEXT_WINDOW = 20  # context window to use
    VECDIM = 300  # WordVec dim to use
    MAX_NORM = 10
    EPOCH_COUNT = 10  # how many epoch to train before rerand neg case
    TRAIN_ROUND = 1000
    DATA_BATCH_SIZE = 128
    # BATCH_SIZE = 8  # how many cases there are in each batch
    DATA_WORKERS = 4
    BATCH_SIZE = 1024 * 24 * 4  # how many cases there are in each batch
    NEG_COUNT = 5  # how many neg case in each case
    NEG_GROUP = 10  # how many group of neg case there are for every postive case
    SHRINK_RATE = 0.75
    SHRINK_WINDOW = 10
    SHRINK_THRESHOLD = 0.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FREQ_HOLD = 1e-6
    FREQ_POW = 0.75  # magic number
    FREQ_MUL = 1000000
    LR_FILE = "./lr.txt"
    LR = get_lr(LR_FILE)
    # LR = 0.0003
    MOMENTUM = 0.75
    DAMPENING = 0.1
    NET_SAVE_FOLDER = "data/nets/Adam/"
    NET_SAVE_PREFIX = "A"
    DATA_FOLDER = "../../download/OANC-GrAF/data"
    DATA_SUFFIX = ".txt"
    DICTIONARY_PATH = "dictionary.txt"
    WORD_COUNT_PATH = "word_count.txt"
    MODIFIED_WORD_COUNT_PATH = "word_count.mod.txt"
    if DEBUG:
        DEVICE = "cpu"
        NEG_GROUP = 1  # how many group of neg case there are for every postive case
        VECDIM = 30  # WordVec dim to use
        NET_SAVE_PREFIX += "-DEBUG"

    NET_SAVE_FILENAME_TEMPLATE = NET_SAVE_FOLDER + NET_SAVE_PREFIX
    # text = """We are about to study the idea of a computational process.
    # Computational processes are abstract beings that inhabit computers.
    # As they evolve, processes manipulate other abstract things called data.
    # The evolution of a process is directed by a pattern of rules
    # called a program. People create programs to direct processes. In effect,
    # we conjure the spirits of the computer with our spells. """.split()

    dataFiles = loadDataFile(DATA_FOLDER, DATA_SUFFIX)
    fileContents = []
    wordset = []
    wordtoid = {}
    idtoword = []
    idContent = []
    wordcnt = collections.Counter()
    modifiedwordcount = {}
    wordBucket = []

    lengthCnt = 0
    print("%10d data file found" % (len(dataFiles)))
    with Timer() as fileLoadTimer:
        for filePath in dataFiles:
            print("Loading From %50s" % (filePath), end="\r")
            with Timer() as timer:
                fileContents.append(loadFile(filePath))
                wordset.extend(getWordSetFromContent(fileContents[-1]))
                lenth = len(fileContents[-1])
            print("Loaded %12d words in %15f sec from file %50s" %
                  (lenth, timer.elapsed, filePath),
                  end="\n")
            lengthCnt += lenth
        wordset = [word for word in set(wordset)]
        wordsetsize = len(wordset)
    print("Loading completed in %15f sec" % (fileLoadTimer.elapsed))
    print("Data Size %20d , Word Set Size %20d,File Count %20d" %
          (lengthCnt, wordsetsize, len(dataFiles)))
    with Timer() as timerFreq1:
        contents = []
        for content in fileContents:
            # wordcnt += collections.Counter(content)
            contents.extend(content)
        rawwordcnt = collections.Counter(contents)
        fileContents = [[
            word if rawwordcnt[word] > lengthCnt * FREQ_HOLD else ":unknow:"
            for word in content
        ] for content in fileContents]
        contents = []
        for content in fileContents:
            # wordcnt += collections.Counter(content)
            contents.extend(content)
        wordset = set(contents)
        wordset = [word for word in set(wordset)]
        wordsetsize = len(wordset)
    print("1st Word Freq Counted %15f sec" % (timerFreq1.elapsed))
    print("Data Size %20d , Word Set Size %20d,File Count %20d" %
          (lengthCnt, wordsetsize, len(dataFiles)))
    with Timer() as timerDictionary:
        wordset.sort()
        wordtoid = {word: i for i, word in enumerate(wordset)}
        idtoword = wordset
        with open(DICTIONARY_PATH, 'w') as f:
            f.write(str(wordtoid))
    print("Dictionary Built in %15f sec" % (timerDictionary.elapsed))
    with Timer() as timerIdContent:
        idContent = [[wordtoid[word] for word in row] for row in fileContents]
        del fileContents
    print("Content Converted in %15f sec" % (timerIdContent.elapsed))
    with Timer() as timerFreq:
        contents = []
        for content in idContent:
            # wordcnt += collections.Counter(content)
            contents.extend(content)
        wordcnt = collections.Counter(contents)
        modifiedwordcount = {
            word: int(((count / lengthCnt)**FREQ_POW) * FREQ_MUL)
            for word, count in wordcnt.items()
        }
        for wordid, count in modifiedwordcount.items():
            wordBucket.extend([wordid for i in range(count)])
        with open(WORD_COUNT_PATH, 'w') as f1:
            f1.write(str(wordcnt))
        with open(MODIFIED_WORD_COUNT_PATH, 'w') as f2:
            f2.write(str(modifiedwordcount))
    print("Word Freq Counted %15f sec" % (timerFreq.elapsed))
    with Timer() as timerNetInit:
        net = WordVec(wordsetsize, VECDIM, CONTEXT_WINDOW, MAX_NORM)
        # optimizer = optim.SGD(net.parameters(),
        #                       lr=LR,
        #                       momentum=MOMENTUM,
        #                       dampening=DAMPENING)
        optimizer = optim.Adam(net.parameters(), lr=LR)
        net.to(DEVICE)
    print("Net created in %15f sec" % (timerNetInit.elapsed))
    # for epoch in range(EPOCH_COUNT):
    print("Start in %s mode" % (DEVICE))

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
    for roundid in range(TRAIN_ROUND):
        with Timer() as timerEpoch:
            random.shuffle(idContent)
            dataBatchCnt = (len(idContent) // DATA_BATCH_SIZE) + 1
            # del rawdata
            for dataBatch in range(dataBatchCnt):
                rawdata = []
                gc.collect()
                l = dataBatch * DATA_BATCH_SIZE
                r = (dataBatch + 1) * DATA_BATCH_SIZE
                with Timer() as timerPostiveSample:
                    pos_finish_cnt = 0
                    for idtext in idContent[l:r]:
                        for i in range(CONTEXT_WINDOW,
                                       len(idtext) - CONTEXT_WINDOW):
                            context = torch.LongTensor([
                                idtext[j]
                                for j in range(i - CONTEXT_WINDOW, i +
                                               CONTEXT_WINDOW + 1) if j != i
                            ])
                            target = torch.LongTensor([idtext[i]])
                            rawdata.append((context, target))
                        pos_finish_cnt += 1
                        print("Postive Simple Progress %15d in %15d" %
                              (pos_finish_cnt, DATA_BATCH_SIZE),
                              end="\r")
                print(
                    "Loaded %20d Postive Simple in %15f sec from dataBatch %3d / %3d round %5d"
                    % (len(rawdata), timerPostiveSample.elapsed, dataBatch,
                       dataBatchCnt, roundid))

                # -------------------------------
                print("resampling for round %d dataBatch %d" %
                      (roundid, dataBatch))
                data = []
                for _ in range(NEG_GROUP):
                    # with Pool(DATA_WORKERS) as pool:
                    data.extend(
                        map(
                            functools.partial(neg_sample, NEG_COUNT,
                                              wordBucket), rawdata))
                # data = NegDataLoader(rawdata, wordBucket, NEG_COUNT, NEG_GROUP)
                # print(1)
                # negdata = next(
                #     iter(
                #         torch.utils.data.DataLoader(data,
                #                                     batch_size=len(data),
                #                                     num_workers=DATA_WORKERS,
                #                                     shuffle=False)))
                # print(2)
                # finaldata = [(negdata[0][i], negdata[1][i], negdata[2][i])
                #              for i in range(len(data))]
                # print(finaldata[0:1])
                # print(3)
                dataloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=BATCH_SIZE,
                    num_workers=DATA_WORKERS,
                    #  shuffle=True)
                    shuffle=False)
                print("resampled for round %d databatch %d" %
                      (roundid, dataBatch))
                for epoch in range(EPOCH_COUNT):
                    for i, data in enumerate(dataloader):
                        # print(len(data[0]))
                        with Timer() as timer:
                            net.zero_grad()
                            loss = net(data[0].to(DEVICE), data[1].to(DEVICE),
                                       data[2].to(DEVICE))
                            loss.backward()
                            accumateloss += loss.item()
                            epochaccloss += loss.item()
                            optimizer.step()
                        timeacc += timer.elapsed
                        if outputcount % OUTPUT_INTERVAL == 0 and outputcount != 0:
                            if outputtitlecount % OUTPUT_TITLE_INTERVAL == 0:
                                print(
                                    "%5s %5s %5s %10s %10s %10s %15s  %15s  %10s %10s %5s %5s %10s"
                                    %
                                    ("round", "epoch", "batch", "loss",
                                     "epochloss", "last_eloss", "imporve_rate",
                                     "last_im_rate", "lr", "momentum", "scnt",
                                     "bcnt", "sec/batch"))
                            curepochloss = epochaccloss / (batchcnt + 1)
                            imporve_rate = (epochloss -
                                            curepochloss) / epochloss
                            print(
                                "%5d %5d %5d %10f %10f %10f %15f%% %15f%% %10f %10f %5d %5d %10f"
                                % (
                                    roundid,
                                    epoch,
                                    i,
                                    accumateloss / OUTPUT_INTERVAL,
                                    curepochloss,
                                    epochloss,
                                    imporve_rate * 100.0,
                                    imporve_rate_by_epoch * 100.0,
                                    optimizer.param_groups[0]['lr'],
                                    #optimizer.param_groups[0]['momentum'],
                                    math.nan,
                                    shrink_cnt,
                                    back_count,
                                    timeacc / OUTPUT_INTERVAL))
                            accumateloss = 0.0
                            timeacc = 0.0
                            outputtitlecount += 1
                        outputcount += 1
                        batchcnt += 1
                    epochcount += 1
                    # if (epochloss - curepochloss)/epochloss < 0.0001
                    curepochloss = epochaccloss / batchcnt
                    imporve_rate_by_epoch = (epochloss -
                                             curepochloss) / epochloss
                    epochloss = curepochloss
                    losses_for_shrink.append(imporve_rate_by_epoch)
                    epochaccloss = 0.0
                    batchcnt = 0
                    # print(losses_for_shrink[-SHRINK_WINDOW:])
                    back_count = [
                        1 if l > 0 else 0
                        for l in losses_for_shrink[-SHRINK_WINDOW:]
                    ].count(0)
                    # print(back_count)
                    if back_count > SHRINK_THRESHOLD * SHRINK_WINDOW:
                        # lr_tim *= SHRINK_RATE
                        losses_for_shrink = []
                        # shrink_cnt += 1
                    LR = get_lr(LR_FILE)
                    adjust_learning_rate(optimizer, epochcount, LR * lr_tim)
                    if epochcount % SAVE_INTERVAL_BY_EPOCH == 0:
                        pass
                del rawdata
        torch.save(net.cpu(),
                   NET_SAVE_FILENAME_TEMPLATE + "-%d.tr" % (epoch + 1))
        net.to(DEVICE)
        print("epoch %15f sec" % (timerEpoch.elapsed))
    torch.save(net.cpu(), NET_SAVE_FILENAME_TEMPLATE + "-Finished.tr")


if __name__ == "__main__":
    main()