import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import time
import copy
import torch
from torch import nn
import pytest
import functools
from deep_rewire import DEEPR, SoftDEEPR, convert, reconvert
from deep_rewire.utils import measure_sparsity
from models import FCN, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("model_class", [FCN, CNN])
def test_backward_pass(model_class):
    model = model_class()
    convert(model)
    model.to(device)
    sample_input = torch.randn(1, *model.input_shape, device=device)
    sample_output = torch.randn(1, *model.output_shape, device=device)
    criterion = torch.nn.MSELoss()
    optimizer = SoftDEEPR(model.parameters(), lr=0.05, l1=0.005)

    initial_params = {name: param.clone() for name, param in model.named_parameters()}

    output = model(sample_input)
    loss = criterion(output, sample_output)
    loss.backward()
  
    for name, param in model.named_parameters():
        if '_signs' in name:
            assert param.grad is None, "A 'sign' parameter has a gradient"
        else:
            assert param.grad is not None, "A normal parameter has no gradient"

    optimizer.step()
     
    for name, param in model.named_parameters():
        if '_sign' in name:
            assert torch.equal(initial_params[name], param), "A 'sign' parameter changed"
        else:
            assert not torch.equal(initial_params[name], param), "A normal parameter didn't change"

@pytest.mark.parametrize("model_class", [FCN, CNN])
def test_overfitting_small_batch(model_class):
    model = model_class()
    convert(model)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = SoftDEEPR(model.parameters(), lr=0.6, l1=0.0005)

    sample_input = torch.randn(10, *model.input_shape, device=device)
    sample_output = torch.randn(10, *model.output_shape, device=device)
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()
    
    reconvert(model)
    assert measure_sparsity(model.parameters()) > 0.7, "SoftDEEPR produced a model which is not sparse enough (below 70%)"
    assert loss.item() < 0.05, "SoftDEEPR couldn't fit the input to the output."

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
        sample_input = torch.randn(32, 28*28, device=device)
        sample_output = torch.randn(32, 10, device=device)
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    return end_time - start_time

def test_training_time():
    set_random_seed(42)

    model = FCN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for _ in range(10):
        sample_input = torch.randn(32, 28*28, device=device)
        sample_output = torch.randn(32, 10, device=device)
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()

    regular_time = profile_optimizer(optimizer, model, criterion)

    model = FCN()
    convert(model)
    model.to(device)
    optimizer = SoftDEEPR(model.parameters(), lr=0.5, l1=0.0005)
    
    for _ in range(10):
        sample_input = torch.randn(32, 28*28, device=device)
        sample_output = torch.randn(32, 10, device=device)
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()

    rewired_time = profile_optimizer(optimizer, model, criterion)

    print(f"Regular SGD time: {regular_time:.4f}s, SoftDEEPR time: {rewired_time:.4f}s")
    assert rewired_time < 7 * regular_time
