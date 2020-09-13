from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import numpy as np
import math

class ReplayBuffer(object):
    def __init__ (self, N_epi=50, N_len=300):
        self.N_epi = N_epi
        self.N_len = N_len
        self.memory = deque(maxlen=self.N_epi)
        
        self.counter = 0

    def reset(self):
        self.current = 0
        self.memory.clear()

    def store(self, epi):
        if len(epi) > self.N_len:
            return 
        self.memory.append(epi)
        self.counter += 1

    def sample(self, N_batch):
        indexs = np.random.randint(low=0, high=self.counter, size=(N_batch,)) % self.N_epi
        ret = []
        for index in indexs:
            ret.append(self.memory[index])
        return ret


        
        
