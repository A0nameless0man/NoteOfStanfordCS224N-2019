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


class Speaker(nn.Module):
    def __init__(self, word_set_size: int, word_dim_length: int,
                 word_dim_max_norm: float):
        super(Speaker, self).__init__()
        self.wordvec = nn.Embedding(word_set_size,
                                    word_dim_length,
                                    max_norm=MAX_NORM)
