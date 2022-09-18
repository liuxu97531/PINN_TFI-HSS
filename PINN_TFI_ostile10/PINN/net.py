import torch
import torch.optim
from collections import OrderedDict
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, seq_net, name='MLP', activation=torch.tanh):
        super().__init__()
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
        self.features = nn.ModuleDict(self.features)
        self.active = activation

        # initial_bias

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_normal_(m.weight, gain=1.0) #ini2
                # torch.nn.init.uniform_(m.weight, a=-0.01, b=0.01) #3
                # torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1) #4
                # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01) #5
                # torch.nn.init.normal_(m.weight, mean=0.0, std=0.1) #6

    def forward(self, x):
        # x = x.view(-1, 2)
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1: break
            i += 1
            x = self.active(x)
        return x

