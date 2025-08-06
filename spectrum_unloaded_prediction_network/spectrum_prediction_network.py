import torch
import torch.nn as nn
from collections import OrderedDict

linear_len = 2048
class ForwardNetMlp(nn.Module):

    def __init__(self):
        super(ForwardNetMlp, self).__init__()

        self.structure_dimension = 4
        self.spectrum_dimension = 1001

        self.n1 = nn.Sequential(OrderedDict({
            'linear1': nn.Linear(self.structure_dimension, linear_len),
            'relu1': nn.ReLU(inplace=True)}))

        self.fc = nn.Sequential()
        for i in range(8):
            self.fc.add_module("linear" + str(i), nn.Linear(linear_len, linear_len))
            self.fc.add_module("relu" + str(i), nn.ReLU(inplace=True))

        self.nn = nn.Sequential(OrderedDict({
            'linear1': nn.Linear(linear_len, self.spectrum_dimension)}))

    def forward(self, x):
        x = self.n1(x)
        x = self.fc(x)
        x = self.nn(x)
        return x


if __name__ == "__main__":
    model = ForwardNetMlp()
    x = torch.randn(32, 4)
    x = model(x)
    print(x)
    print("welcome Teddy")
