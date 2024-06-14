import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import copy
from torch import nn
from src import DEEPR, SoftDEEPR, convert_to_deep_rewireable, convert_from_deep_rewireable
from src.converter import NonTrainableParameter
from src.utils import measure_sparsity
import pytest

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
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

def test_non_trainable_parameter():
    # Create a NonTrainableParameter
    param = NonTrainableParameter(torch.tensor([1.0, 2.0, 3.0]))

    # Assert that requires_grad is False
    assert param.requires_grad == False, "NonTrainableParameter should have requires_grad set to False"

    # Try setting requires_grad to True and assert it is still False
    param.requires_grad = True
    assert param.requires_grad == False, "NonTrainableParameter should not allow requires_grad to be set to True"

    # Try setting requires_grad to False and assert it is still False
    param.requires_grad = False
    assert param.requires_grad == False, "NonTrainableParameter should keep requires_grad set to False"

def test_forward_pass():
    model = FCN()
    convert_to_deep_rewireable(model)
    sample_input = torch.randn(1, 28*28)
    output = model(sample_input)
    assert output.shape == (1, 10)

def test_loss_calculation():
    model = FCN()
    convert_to_deep_rewireable(model)
    sample_input = torch.randn(1, 28*28)
    sample_output = torch.randn(1, 10)
    criterion = torch.nn.MSELoss()
    output = model(sample_input)
    loss = criterion(output, sample_output)
    assert loss.item() > 0

def test_backward_pass():
    model = FCN()
    convert_to_deep_rewireable(model)
    sample_input = torch.randn(1, 28*28)
    sample_output = torch.randn(1, 10)
    criterion = torch.nn.MSELoss()
    optimizer = SoftDEEPR(model.parameters(), lr=0.05, l1=0.005)

    output = model(sample_input)
    loss = criterion(output, sample_output)
    loss.backward()
  
    # everything but signs should have gradient  
    for name, param in model.named_parameters():
        if '_signs' in name:
            assert param.grad is None
        else:
            assert param.grad is not None


def test_parameter_updates():
    model = FCN()
    convert_to_deep_rewireable(model)
    sample_input = torch.randn(1, 28*28)
    sample_output = torch.randn(1, 10)
    criterion = torch.nn.MSELoss()
    optimizer = SoftDEEPR(model.parameters(), lr=0.05, l1=0.005)

    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    output = model(sample_input)
    loss = criterion(output, sample_output)
    loss.backward()
    optimizer.step()
     
    for name, param in model.named_parameters():
        if '_sign' in name:
            assert torch.equal(initial_params[name], param)
        else:
            assert not torch.equal(initial_params[name], param)


def test_overfitting_small_batch():
    model = FCN()
    convert_to_deep_rewireable(model)
    criterion = torch.nn.MSELoss()
    optimizer = SoftDEEPR(model.parameters(), lr=0.5, l1=0.0005)

    sample_input = torch.randn(10, 28*28)
    sample_output = torch.randn(10, 10)
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()
    
    convert_from_deep_rewireable(model)
    assert measure_sparsity(model.parameters()) > 0.7
    assert loss.item() < 0.1


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def profile_optimizer(optimizer, model, criterion, epochs=100):
    start_time = time.time()
    for epoch in range(epochs):
        sample_input = torch.randn(32, 28*28)
        sample_output = torch.randn(32, 10)
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    return end_time - start_time

def test_training_time():
    set_random_seed(42)

    # Standard SGD
    model = FCN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Warm-up
    for _ in range(10):
        sample_input = torch.randn(32, 28*28)
        sample_output = torch.randn(32, 10)
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()

    regular_time = profile_optimizer(optimizer, model, criterion)
    
    #print(prof.key_averages().table(sort_by="cpu_time_total"))

    # SoftDEEPR
    model = FCN()
    convert_to_deep_rewireable(model)
    optimizer = SoftDEEPR(model.parameters(), lr=0.5, l1=0.0005)
    
    # Warm-up
    for _ in range(10):
        sample_input = torch.randn(32, 28*28)
        sample_output = torch.randn(32, 10)
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()

    rewired_time = profile_optimizer(optimizer, model, criterion)
    
    #print(prof.key_averages().table(sort_by="cpu_time_total"))

    print(f"Regular SGD time: {regular_time:.4f}s, SoftDEEPR time: {rewired_time:.4f}s")
    assert rewired_time < 7 * regular_time
