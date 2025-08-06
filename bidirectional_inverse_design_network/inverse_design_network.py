import torch
from torch import nn
from collections import OrderedDict

linear_len = 512

class InverseNet(torch.nn.Module):
    def __init__(self):
        super(InverseNet, self).__init__()
        self.n1 = nn.Sequential(OrderedDict({
            'linear': nn.Linear(5, linear_len),
            'relu': nn.ReLU(inplace=True)}))

        self.fc = nn.Sequential()
        for i in range(4):
            self.fc.add_module("linear" + str(i), nn.Linear(linear_len, linear_len))
            self.fc.add_module("relu" + str(i), nn.ReLU(inplace=True))

        self.nn = nn.Sequential(OrderedDict({
            'linear1': nn.Linear(linear_len, 4),
            'sigmoid': nn.Sigmoid()}))

    def forward(self, x):

        x = self.n1(x)
        x = self.fc(x)
        x = self.nn(x)
        return x
