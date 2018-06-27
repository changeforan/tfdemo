import numpy as np

class Quadratic:
# f(x) = ax^2+b
    def __init__(self,
                 a,
                 b,
                 start,
                 end,
                 num):
        self.x_ = np.linspace(start, end, num)[:, np.newaxis]
        self.y_ = np.square(self.x_) * a + b


class Recursion:
# a_0 = 0
# a_(n+1) = {a_(n)^2 + 1} / 2
    def __init__(self, num):
        self.a = None
        self.b = 0
        self.num = num
        self.x_ = np.linspace(0, num - 1, num)[:, np.newaxis]
        self.y_ = list(self)

    def next(self):
        self.num -= 1
        if self.num >= 0:
            self.a, self.b = self.b, (self.b ** 2 + 1.) / 2
            return [self.a]
        else:
            raise StopIteration

    def __iter__(self):
        return self