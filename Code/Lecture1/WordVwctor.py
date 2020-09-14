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
from pathlib import Path
import os
import gc
import functools
import datetime
from tensorboardX import SummaryWriter
import sys

DATE_STRING = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
# DATE_STRING = datetime.datetime.now().replace(microsecond=0).isoformat()


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
    DATA_BATCH_SIZE = 32
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
    NET_SAVE_FOLDER = "results/Adam/%s/net" % (DATE_STRING)
    LOG_SAVE_FOLDER = "results/Adam/%s/log" % (DATE_STRING)
    TEXT_SAVE_FOLDER = "results/Adam/%s/text" % (DATE_STRING)
    NET_SAVE_PREFIX = "net-%s.tr"
    DATA_FOLDER = "../../download/OANC-GrAF/data"
    DATA_SUFFIX = ".txt"
    DICTIONARY_PATH = TEXT_SAVE_FOLDER + "/dictionary.txt"
    WORD_COUNT_PATH = TEXT_SAVE_FOLDER + "/word_count.txt"
    MODIFIED_WORD_COUNT_PATH = TEXT_SAVE_FOLDER + "/word_count.mod.txt"
    if DEBUG:
        DEVICE = "cpu"
        NEG_GROUP = 1  # how many group of neg case there are for every postive case
        VECDIM = 30  # WordVec dim to use
        NET_SAVE_PREFIX += "-DEBUG"

    NET_SAVE_FILENAME_TEMPLATE = NET_SAVE_FOLDER + "/" + NET_SAVE_PREFIX

    for folder in [NET_SAVE_FOLDER, LOG_SAVE_FOLDER, TEXT_SAVE_FOLDER]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(LOG_SAVE_FOLDER)
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
    writer.add_text("info/PyTorch version", torch.__version__)
    writer.add_text("info/Cuda version", str(torch.version.cuda))
    writer.add_text("info/Cuda Device Count", str(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(i)
        writer.add_text(
            "info/Cuda device/ %d : %s" % (i, info.name),
            "CUDA CC %d.%d ;MEM %dMB ;MultiProcessorCount %d" %
            (info.major, info.minor, info.total_memory /
             (1024**2), info.multi_processor_count))
    writer.add_text("info/Python version", sys.version)
    writer.add_text("info/Data Folder", DATA_FOLDER)
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

    epochcount = 0
    epochaccloss = 0.0
    negGroupAccloss = 0.0
    batchcnt = 0
    lr_tim = 1.
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
                    data.extend(
                        map(
                            functools.partial(neg_sample, NEG_COUNT,
                                              wordBucket), rawdata))
                dataloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=BATCH_SIZE,
                    num_workers=DATA_WORKERS,
                    #  shuffle=True)
                    shuffle=False)
                print("resampled for round %d databatch %d" %
                      (roundid, dataBatch))
                negGroupAccloss = 0.0
                negGroupMaxloss = 0.0
                batchOfCurNegSample = 0
                print("%10s %10s %10s" % ("epochloss", "lr", "epochcnt"))
                for epoch in range(EPOCH_COUNT):
                    epochaccloss = 0.0
                    batchOfCurEpoch = 0
                    for i, data in enumerate(dataloader):
                        with Timer() as batchTimer:
                            net.zero_grad()
                            loss = net(data[0].to(DEVICE), data[1].to(DEVICE),
                                       data[2].to(DEVICE))
                            loss.backward()
                            epochaccloss += loss.item()
                            optimizer.step()
                            #----------------------------------------------------------------
                            writer.add_scalars("loss", {"batch": loss.item()},
                                               batchcnt)
                            #----------------------------------------------------------------
                        #----------------------------------------------------------------
                        writer.add_scalars("time",
                                           {"batch": batchTimer.elapsed},
                                           batchcnt)
                        #----------------------------------------------------------------
                        batchcnt += 1
                        batchOfCurEpoch += 1
                        batchOfCurNegSample += 1
                    curepochloss = epochaccloss / batchOfCurEpoch
                    epochloss = curepochloss
                    negGroupAccloss += epochloss
                    negGroupMaxloss = max(negGroupMaxloss, epochloss)
                    #----------------------------------------------------------------
                    writer.add_scalars("loss", {"epoch": epochloss},
                                       batchcnt - batchOfCurEpoch // 2)
                    # writer.add_scalar("lr", LR, batchcnt)
                    print("%10f %10f %10d" % (epochloss, LR, epochcount))
                    #----------------------------------------------------------------
                    epochcount += 1
                    LR = get_lr(LR_FILE)
                    adjust_learning_rate(optimizer, epochcount, LR * lr_tim)

                #----------------------------------------------------------------
                writer.add_scalars(
                    "loss", {
                        "neg_avg": negGroupAccloss / EPOCH_COUNT,
                        "neg_max": negGroupMaxloss
                    }, batchcnt - batchOfCurNegSample // 2)
                #----------------------------------------------------------------
        #----------------------------------------------------------------
        writer.add_scalar("epoch time", timerEpoch.elapsed, roundid)
        #----------------------------------------------------------------

        torch.save(net.cpu(),
                   NET_SAVE_FILENAME_TEMPLATE % ("%d" % (roundid + 1)))
        net.to(DEVICE)
        print("epoch %15f sec" % (timerEpoch.elapsed))
    torch.save(net.cpu(), NET_SAVE_FILENAME_TEMPLATE % ("Finished"))


if __name__ == "__main__":
    main()