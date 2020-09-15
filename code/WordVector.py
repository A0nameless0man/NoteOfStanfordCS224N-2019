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
import functools
# from tensorboardX import SummaryWriter

from torch.utils.tensorboard import SummaryWriter
from code.Tools.Time import Timer, getTimeString
from code.Tools.ConfigLogger import LogConfig
from code.Tools.GpuTempSensor import get_gpu_tem
from code.Tools.Mkdir import mkdirs
from code.Tools.Ls import ls
from code.Tools.LoggedType import LoggedFloat
from code.Tools.Counter import Counter

# DATE_STRING = datetime.datetime.now().replace(microsecond=0).isoformat()


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
    return (case[0], case[1],
            torch.LongTensor(random.sample(wordBucket, NEG_COUNT)))


def build_postive_samples(CONTEXT_WINDOW, raw):
    samples = []
    for i in range(CONTEXT_WINDOW, len(raw) - CONTEXT_WINDOW):
        context = torch.LongTensor([
            raw[j] for j in range(i - CONTEXT_WINDOW, i + CONTEXT_WINDOW + 1)
            if j != i
        ])
        target = torch.LongTensor([raw[i]])
        samples.append((context, target))
    return samples


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
    DATE_STRING = getTimeString()

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
    LR = 0.001
    MOMENTUM = 0.75
    DAMPENING = 0.1
    SAVE_FOLDER_PREFIX = "results/WordVector/Adam/"
    NET_SAVE_FOLDER = SAVE_FOLDER_PREFIX + "%s/net" % (DATE_STRING)
    LOG_SAVE_FOLDER = SAVE_FOLDER_PREFIX + "%s/log" % (DATE_STRING)
    TEXT_SAVE_FOLDER = SAVE_FOLDER_PREFIX + "%s/text" % (DATE_STRING)
    NET_SAVE_PREFIX = "net-%s.tr"
    DATA_FOLDER = "./download/OANC-GrAF/data"
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
    mkdirs([NET_SAVE_FOLDER, LOG_SAVE_FOLDER, TEXT_SAVE_FOLDER])
    writer = SummaryWriter(LOG_SAVE_FOLDER)
    dataFiles = ls(DATA_FOLDER, DATA_SUFFIX)
    fileContents = []
    wordset = []
    wordtoid = {}
    idtoword = []
    idContent = []
    wordcnt = collections.Counter()
    modifiedwordcount = {}
    wordBucket = []
    lengthCnt = 0
    LogConfig(locals(), writer)

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
    batchcnt = 0
    batchCounter = Counter()
    epochCounter = Counter()
    for roundid in range(TRAIN_ROUND):
        with Timer() as timerEpoch:
            random.shuffle(idContent)
            dataBatchCnt = (len(idContent) // DATA_BATCH_SIZE) + 1
            # del rawdata
            for dataBatch in range(dataBatchCnt):
                rawdata = []
                l = dataBatch * DATA_BATCH_SIZE
                r = (dataBatch + 1) * DATA_BATCH_SIZE
                with Timer() as timerPostiveSample:
                    pos_finish_cnt = 0
                    for idtext in idContent[l:r]:
                        rawdata.extend(
                            build_postive_samples(CONTEXT_WINDOW, idtext))
                        pos_finish_cnt += 1
                        # print("Postive Simple Progress %15d in %15d" %
                        #       (pos_finish_cnt, DATA_BATCH_SIZE),
                        #       end="\r")
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
                negGrouploss = LoggedFloat()
                negSampleLap = batchCounter.lap()
                print("%10s %10s %10s" % ("epochloss", "lr", "epochcnt"))
                for epoch in range(EPOCH_COUNT):
                    epochLap = batchCounter.lap()
                    epochloss = LoggedFloat()
                    for i, data in enumerate(dataloader):
                        with Timer() as batchTimer:
                            net.zero_grad()
                            loss = net(data[0].to(DEVICE), data[1].to(DEVICE),
                                       data[2].to(DEVICE))
                            loss.backward()
                            optimizer.step()

                            epochloss.update(loss.item())
                            #----------------------------------------------------------------
                            writer.add_scalars("loss", {"batch": loss.item()},
                                               batchCounter.value)
                            #----------------------------------------------------------------
                        #----------------------------------------------------------------
                        writer.add_scalars("time",
                                           {"batch": batchTimer.elapsed},
                                           batchCounter.value)
                        #----------------------------------------------------------------
                        batchCounter.step()
                    #----------------------------------------------------------------
                    writer.add_scalars(
                        "loss", {
                            "epoch_avg": epochloss.avg,
                            "epoch_max": epochloss.max,
                            "epoch_min": epochloss.min
                        }, batchCounter.mean(epochLap))
                    writer.add_scalars("temp", {"GPU": get_gpu_tem()},
                                       epochCounter.value)
                    print("%10f %10f %10d" %
                          (epochloss.avg, LR, epochCounter.value))
                    #----------------------------------------------------------------
                    negGrouploss.update(epochloss.avg)
                    epochCounter.step()
                #----------------------------------------------------------------
                writer.add_scalars(
                    "loss", {
                        "neg_avg": negGrouploss.avg,
                        "neg_max": negGrouploss.max,
                        "neg_min": negGrouploss.min
                    }, batchCounter.mean(negSampleLap))
                #----------------------------------------------------------------
        #----------------------------------------------------------------
        writer.add_scalar("epoch time", timerEpoch.elapsed, roundid)
        #----------------------------------------------------------------

        torch.save(net.cpu(), NET_SAVE_FILENAME_TEMPLATE % ("%d" % (roundid)))
        net.to(DEVICE)
        print("epoch %15f sec" % (timerEpoch.elapsed))
    torch.save(net.cpu(), NET_SAVE_FILENAME_TEMPLATE % ("Finished"))


if __name__ == "__main__":
    main()