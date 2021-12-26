import numpy as np
from abc import ABC, abstractmethod
from scipy.special import expit

class ANeuron(ABC):
    @abstractmethod
    def f(self, x):
        pass

    def df(self, x):
        pass

class NeuronSigm(ANeuron):
    def __init__(self, y):
        self.y = y

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, x):
        return expit(self.f(x) * (1 - self.f(x)))


class NeuronTang(ANeuron):
    def __init__(self, y):
        self.y = y

    def f(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def df(self, x):
        return expit(1 - self.f(x) ** 2)

W1 = np.array([[-0.2, 0.2, -0.3], [0.2, -0.1, -0.4]])
W2 = np.array([0.1, 0.3])


def forward(inpt, h):
        sum = np.dot(W1, inpt)
        if h == 1:
            act = NeuronTang(sum)
        if h == 2:
            act = NeuronSigm(sum)
        out = np.array([act.f(x) for x in sum])
        sum = np.dot(W2, out)
        y = act.f(sum)
        return (y, out)

def training(sample, h):
        global W2, W1
        step = 0.01
        N = 11000
        count = len(sample)
        for k in range(N):
            x = sample[np.random.randint(0, count)]
            y, out = forward(x[0:3],h)
            e = y - x[-1]
            if h == 1:
                act = NeuronTang(y)
            if h == 2:
                act = NeuronSigm(y)
            gradient1 = e * act.df(y)
            W2[0] = W2[0] - step * gradient1 * out[0]
            W2[1] = W2[1] - step * gradient1 * out[1]
            act = NeuronTang(out)
            gradient2 = W2 * gradient1 * act.df(out)

            W1[0, :] = W1[0, :] - np.array(x[0:3]) * gradient2[0] * step
            W1[1, :] = W1[1, :] - np.array(x[0:3]) * gradient2[1] * step

sample = [(-1, -1, -1, -1),
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, -1, 1, 1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]

print("Введите функцию активации 1 - логистическая 2 - тангенсальная")
l = int(input())
training(sample,l)
pred = np.array([-1,1,1])
y, out = forward(pred,l)
print("Конечные весовые коэффициенты W1: ", W1)
print("Конечные весовые коэффициенты W2: ", W2)
print(f"Выходное значение: {y} => {pred[-1]}")

