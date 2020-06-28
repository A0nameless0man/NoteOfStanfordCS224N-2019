from __future__ import print_function
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class WordVec(nn.Module):
    def __init__(self,size:int,length:int,window:int):
        super(WordVec, self).__init__()
        self.vecsize = size
        self.dirlength = length
        self.window = window
        self.wordvec = nn.Embedding(length, size)
        self.contextVec = nn.Embedding(length, size)
        
    def forward(self, text):
        contextMatrix = contextVec(text)
        wordMatrix = wordvec(text)
        

CONTEXT_WINDOW = 2
text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells. """.split()

wordset = set(text)
wordsetsize = len(wordset)

wordtoid = {word: i for i, word in enumerate(wordset)}

data = []
for i in range(CONTEXT_WINDOW, len(text) - CONTEXT_WINDOW):
    context = [wordtoid[word] for word in ([text[j] for j in range(i - CONTEXT_WINDOW, i + CONTEXT_WINDOW + 1) if j != i])]
    target = wordtoid[text[i]]
    # print((context, target))
    data.append((context, target))
print(data)
# for word in text:
    # print(word+":"+str(wordtoid[word]))