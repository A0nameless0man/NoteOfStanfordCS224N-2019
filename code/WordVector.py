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
from code.Module.WordToVec import WordVec
# from code.Module.WordToVec import neg_sample
from code.Module.WordToVec import build_postive_samples
from code.Module.WordToVec import build_word_bucket
from code.Module.WordToVec import batch_neg_sample
from code.DataSet.TextFile import loadDataSet

from torch.utils.tensorboard import SummaryWriter
from code.Tools.Time import Timer, getTimeString
from code.Tools.ConfigLogger import LogConfig
from code.Tools.GpuTempSensor import get_gpu_tem
from code.Tools.FileOp import mkdirs, ls, print_to_file
from code.Tools.LoggedType import LoggedFloat
from code.Tools.Counter import Counter

# DATE_STRING = datetime.datetime.now().replace(microsecond=0).isoformat()


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
    SAVE_FOLDER_PREFIX = "results/WordVector/"
    NET_SAVE_FOLDER = SAVE_FOLDER_PREFIX + "%s/net" % (DATE_STRING)
    LOG_SAVE_FOLDER = SAVE_FOLDER_PREFIX + "%s/log" % (DATE_STRING)
    TEXT_SAVE_FOLDER = SAVE_FOLDER_PREFIX + "%s/text" % (DATE_STRING)
    NET_SAVE_PREFIX = "net-%s.tr"
    DATA_FOLDER = "./download/OANC-GrAF/data"
    DATA_SUFFIX = ".txt"
    DICTIONARY_PATH = TEXT_SAVE_FOLDER + "/dictionary.txt"
    WORD_FREQ_PATH = TEXT_SAVE_FOLDER + "/word_freq.txt"
    # MODIFIED_WORD_COUNT_PATH = TEXT_SAVE_FOLDER + "/word_count.mod.txt"
    if DEBUG:
        DEVICE = "cpu"
        NEG_GROUP = 1  # how many group of neg case there are for every postive case
        VECDIM = 30  # WordVec dim to use
        NET_SAVE_PREFIX += "-DEBUG"

    NET_SAVE_FILENAME_TEMPLATE = NET_SAVE_FOLDER + "/" + NET_SAVE_PREFIX

    mkdirs([NET_SAVE_FOLDER, LOG_SAVE_FOLDER, TEXT_SAVE_FOLDER])
    writer = SummaryWriter(LOG_SAVE_FOLDER)

    with Timer() as timerLoad:
        idContent, length, wordtoid, idtoword, word_freq, wordset_size = loadDataSet(
            DATA_FOLDER, FREQ_HOLD, ":unk:")
        wordBucket = build_word_bucket(word_freq, FREQ_POW, FREQ_MUL)
    print("load %d files in %f seconds" % (len(idContent), timerLoad.elapsed))
    LogConfig(locals(), writer)
    print_to_file(DICTIONARY_PATH, str(wordtoid))
    print_to_file(WORD_FREQ_PATH, str(word_freq))
    # exit(0)
    with Timer() as timerNetInit:
        net = WordVec(wordset_size, VECDIM, CONTEXT_WINDOW, MAX_NORM)
        # optimizer = optim.SGD(net.parameters(),
        #                       lr=LR,
        #                       momentum=MOMENTUM,
        #                       dampening=DAMPENING)
        optimizer = optim.Adam(net.parameters(), lr=LR)
        net.to(DEVICE)
    print("Net created in %15f sec" % (timerNetInit.elapsed))
    # for epoch in range(EPOCH_COUNT):
    print("Start in %s mode" % (DEVICE))

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
                extend = data.extend
                for _ in range(NEG_GROUP):
                    extend(batch_neg_sample(NEG_COUNT, wordBucket, rawdata))
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
                print(
                    "%10s %10s %10s %10s %10s" %
                    ("epochloss", "epochloss", "epochloss", "lr", "epochcnt"))
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
                    print("%10f %10f %10f %10f %10d" %
                          (epochloss.avg, epochloss.min, epochloss.max, LR,
                           epochCounter.value))
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