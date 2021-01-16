import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(#(4,19,19)
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#(64,19,19)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(64,9,9)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(#(16,10,10)
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#(128,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(128,4,4)
        )
        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
