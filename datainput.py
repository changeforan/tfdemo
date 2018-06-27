import numpy as np

class DataInput:

    def __init__(self, f):
        self.x_ = f.x_
        self.y_ = f.y_

    def next_batch(self, n):
        if n > len(self.x_):
            n = len(self.x_)
        batch_index = np.random.choice(len(self.x_), size=n, replace=False)
        batch_x = [self.x_[i] for i in batch_index]
        batch_y = [self.y_[i] for i in batch_index]
        return batch_x, batch_y


