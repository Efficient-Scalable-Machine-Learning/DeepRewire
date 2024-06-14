import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import copy
from torch import nn
from src import convert_to_deep_rewireable, convert_from_deep_rewireable
from src.utils import measure_sparsity
import pytest

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512, bias=False)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def test_conversion_FCN():
    rewired_model = FCN()
    standard_model = copy.deepcopy(rewired_model)
    convert_to_deep_rewireable(rewired_model)
    a = set(rewired_model.state_dict().keys())
    b = set(standard_model.state_dict().keys())
    assert a != b
    convert_from_deep_rewireable(rewired_model)
    a = set(rewired_model.state_dict().keys())
    b = set(standard_model.state_dict().keys())
    assert a == b
    s1 = measure_sparsity(rewired_model.parameters())
    s2 = measure_sparsity(standard_model.parameters())
    assert s1 > s2

def test_conversion_CNN():
    rewired_model = CNN()
    standard_model = copy.deepcopy(rewired_model)
    convert_to_deep_rewireable(rewired_model)
    a = set(rewired_model.state_dict().keys())
    b = set(standard_model.state_dict().keys())
    assert a != b
    convert_from_deep_rewireable(rewired_model)
    a = set(rewired_model.state_dict().keys())
    b = set(standard_model.state_dict().keys())
    assert a == b
    s1 = measure_sparsity(rewired_model.parameters())
    s2 = measure_sparsity(standard_model.parameters())
    assert s1 > s2
