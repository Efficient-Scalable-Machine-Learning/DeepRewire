import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import copy
from torch import nn
from deep_rewire import DEEPR, convert, reconvert
from deep_rewire.utils import measure_sparsity
import pytest
import functools
from models import FCN, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def verify_number_of_connections(optimizer):
    active_connections = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad == False:
                continue
            active_connections += (p.data >= 0).sum().item()
    assert active_connections == optimizer.nc, f"Expected {optimizer.nc} active connections, found {active_connections}"

def test_number_of_connections_init():
    model = FCN()
    param_total = sum(p.numel() for p in model.parameters())
    nc = int(param_total * 0.3)
    convert(model)
    sample_input = torch.randn(1, *model.input_shape)
    sample_output = torch.randn(1, *model.output_shape)
    criterion = torch.nn.MSELoss()
    optimizer = DEEPR(model.parameters(), nc=nc, lr=0.05, l1=0.005)
    verify_number_of_connections(optimizer)

    reconvert(model)
    assert sum(torch.count_nonzero(p) for p in model.parameters()) <= nc

def test_number_of_connections_step():
    model = FCN()
    param_total = sum(p.numel() for p in model.parameters())
    nc = int(param_total * 0.3)
    convert(model)
    model.to(device)
    sample_input = torch.randn(1, *model.input_shape).to(device)
    sample_output = torch.randn(1, *model.output_shape).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = DEEPR(model.parameters(), nc=nc, lr=0.05, l1=0.005)

    output = model(sample_input)
    loss = criterion(output, sample_output)
    loss.backward()
    optimizer.step()
    verify_number_of_connections(optimizer)
    reconvert(model)
    assert sum(torch.count_nonzero(p) for p in model.parameters() ) <= nc
 

@pytest.mark.parametrize("model_class", [FCN, CNN])
def test_backward_pass(model_class):
    model = model_class()
    param_total = sum(p.numel() for p in model.parameters())
    sparsity = 0.7
    nc = int(param_total * (1 - sparsity))
    convert(model)
    model.to(device)
    sample_input = torch.randn(1, *model.input_shape).to(device)
    sample_output = torch.randn(1, *model.output_shape).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = DEEPR(model.parameters(), nc=nc, lr=0.05, l1=0.005)

    output = model(sample_input)
    loss = criterion(output, sample_output)
    loss.backward()
  
    # everything but signs should have gradient  
    for name, param in model.named_parameters():
        if '_signs' in name:
            assert param.grad is None
        else:
            assert param.grad is not None


@pytest.mark.parametrize("model_class", [FCN, CNN])
def test_parameter_updates(model_class):
    model = model_class()
    param_total = sum(p.numel() for p in model.parameters())
    sparsity = 0.7
    nc = int(param_total * (1 - sparsity))
 
    convert(model)
    model.to(device)
    sample_input = torch.randn(1, *model.input_shape).to(device)
    sample_output = torch.randn(1, *model.output_shape).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = DEEPR(model.parameters(), nc=nc, lr=0.05, l1=0.005)

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

@pytest.mark.parametrize("model_class", [FCN, CNN])
def test_overfitting_small_batch(model_class):
    model = model_class()
    param_total = sum(p.numel() for p in model.parameters())
    sparsity = 0.7
    nc = int(param_total * (1 - sparsity))
    convert(model)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = DEEPR(model.parameters(), nc=nc, lr=0.6, l1=0.0005)

    sample_input = torch.randn(10, *model.input_shape).to(device)
    sample_output = torch.randn(10, *model.output_shape).to(device)
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()
    
    reconvert(model)
    assert measure_sparsity(model.parameters()) > sparsity
    assert loss.item() < 0.05

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
        sample_input = torch.randn(32, 28*28).to(device)
        sample_output = torch.randn(32, 10).to(device)
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
    model = FCN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Warm-up
    for _ in range(10):
        sample_input = torch.randn(32, 28*28).to(device)
        sample_output = torch.randn(32, 10).to(device)
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()

    regular_time = profile_optimizer(optimizer, model, criterion)
    
    #print(prof.key_averages().table(sort_by="cpu_time_total"))

    # DEEPR
    model = FCN()
    param_total = sum(p.numel() for p in model.parameters())
    sparsity = 0.9
    nc = int(param_total * (1 - sparsity))

    convert(model)
    model.to(device)
    optimizer = DEEPR(model.parameters(), nc=nc, lr=0.5, l1=0.0005)
    
    # Warm-up
    for _ in range(10):
        sample_input = torch.randn(32, 28*28).to(device)
        sample_output = torch.randn(32, 10).to(device)
        optimizer.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_output)
        loss.backward()
        optimizer.step()

    rewired_time = profile_optimizer(optimizer, model, criterion)
    
    #print(prof.key_averages().table(sort_by="cpu_time_total"))

    print(f"Regular SGD time: {regular_time:.4f}s, DEEPR time: {rewired_time:.4f}s")
    assert rewired_time < 7 * regular_time
