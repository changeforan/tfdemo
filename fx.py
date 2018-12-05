import numpy as np


# LPCNet 18-band Bark and LPC
class LPCNet:
    def __init__(self):
        feats = np.load('D:/LPC/feats_34/feats_34.npy')
        self.x_ = feats[:, :18]
        self.y_ = feats[:, -16:]


# f(x) = ax^2+b
class Quadratic:
    def __init__(self,
                 a,
                 b,
                 start,
                 end,
                 num):
        self.x_ = np.linspace(start, end, num)[:, np.newaxis]
        self.y_ = np.square(self.x_) * a + b


# a_0 = 0
# a_(n+1) = {a_(n)^2 + 1} / 2
class Recursion:
    def __init__(self, num):
        self.a = None
        self.b = 0
        self.num = num
        self.x_ = np.linspace(0, num - 1, num)[:, np.newaxis]
        self.y_ = list(self)

    def __next__(self):
        self.num -= 1
        if self.num >= 0:
            self.a, self.b = self.b, (self.b ** 2 + 1.) / 2
            return [self.a]
        else:
            raise StopIteration

    def __iter__(self):
        return self


# Pi(x)
# The number of primes not exceeding x
class PiFunction:
    def __init__(self, x):
        self.x_ = np.linspace(0, x, x + 1)[:, np.newaxis]
        self.y_ = pi_arr(x)


def pi_arr(n):
    arr = [i for i in range(n + 1)][2:]
    for i in range(int(n ** 0.5) + 1):
        if arr[i] > 0:
            j = 2
            while j * arr[i] <= n:
                arr[j * arr[i] - 2] = 0
                j += 1
    pi = [[0], [0]]
    count = 0
    for i in range(0, n - 1):
        if arr[i] > 0:
            count += 1
        pi.append([count / 1.])
    return pi
