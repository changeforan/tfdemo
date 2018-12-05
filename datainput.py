import numpy as np


class DataInput:

    def __init__(self, f):
        self.x_ = f.x_
        self.y_ = f.y_
        self.len = len(self.x_)

    def next_batch(self, n):
        if n > self.len:
            n = self.len
        batch_index = np.random.choice(self.len, size=n, replace=True)
        batch_x = [self.x_[i] for i in batch_index]
        batch_y = [self.y_[i] for i in batch_index]
        return batch_x, batch_y


