from collections import deque
import random
from torch.nn.functional import threshold
import numpy as np



class ReplayBuffer(object):
    def __init__(self, capacity, threshold):
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.current = 0
        self.available = False
        self.memory = deque(maxlen = capacity)
        self.threshold = threshold

    def push(self, data):
        self.memory.append(data)
        self.current = (self.current + 1) % self.capacity
        if len(self.memory) >= self.threshold:
            self.available = True

    def sample(self, num):
        if not self.available:
            return []

        return random.sample(self.memory, num)        

    def clear(self):
        self.current = 0
        self.available = False
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, index):
        return self.memory[index]
    


if __name__ == "__main__":
    rb = ReplayBuffer(capacity=3, threshold=3)
    rb.push([1,2,3])
    rb.push([1243])
    rb.push([3,4,6])
    rb.push([2441])
    rb.push(23123)

    print(rb.sample(3))
    print(rb.sample(1))

