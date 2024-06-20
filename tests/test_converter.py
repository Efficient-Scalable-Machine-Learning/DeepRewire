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
        self.input_shape = (28*28,)
        self.output_shape = (10,)
        self.fc1 = nn.Linear(28*28, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, 10, bias=False)
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
        self.input_shape = (3, 32, 32,)
        self.output_shape = (10,)
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

@pytest.mark.parametrize("model_class", [FCN, CNN])
def test_parameters(model_class):
    rewired_model = model_class()
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

@pytest.mark.parametrize("model_class", [FCN, CNN])
@pytest.mark.parametrize("handle_biases", ['ignore', 'as_connections', 'second_bias'])
def test_reconversion(model_class, handle_biases):
    for i in range(10):
        with torch.no_grad():
            model = model_class()
            model.eval()
            convert_to_deep_rewireable(model, handle_biases=handle_biases)
            inpt = torch.rand((1, *model.input_shape))
            out_pre_reconversion = model(inpt)
            convert_from_deep_rewireable(model)
            out_post_reconversion = model(inpt)
        assert torch.equal(out_pre_reconversion, out_post_reconversion)


@pytest.mark.parametrize("model_class", [FCN, CNN])
@pytest.mark.parametrize("handle_biases", ['ignore', 'as_connections', 'second_bias'])
def test_conversion(model_class, handle_biases):
    for i in range(10):
        with torch.no_grad():
            model = model_class()
            model.eval()
            inpt = torch.rand((1, *model.input_shape))
            out_pre_conversion = model(inpt)
            convert_to_deep_rewireable(model, handle_biases=handle_biases, keep_signs=True)
            out_post_conversion = model(inpt)
        assert torch.equal(out_pre_conversion, out_post_conversion)


@pytest.mark.parametrize("model_class", [FCN, CNN])
@pytest.mark.parametrize("handle_biases", ['ignore', 'as_connections', 'second_bias'])
def test_active_probability(model_class, handle_biases):
    for s in range(11):
        connectivity = 1 - s / 10
        sparsities = []
        for i in range(20):
            with torch.no_grad():
                model = model_class()
                convert_to_deep_rewireable(model, handle_biases=handle_biases, active_probability=connectivity)
                convert_from_deep_rewireable(model)
                sparsities.append(measure_sparsity(model.parameters()))
        sparsity = sum(sparsities)/len(sparsities)
        assert pytest.approx(connectivity, abs=0.001) == 1.0 - sparsity
