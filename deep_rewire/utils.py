"""
provides some useful functions
"""

import torch

def check_any_parameter_in_network(model, parameter_name):
    for name, param in model.named_parameters():
        if name == parameter_name:
            return True
    return False

def measure_sparsity(parameters, threshold=0):
    """
    Measures sparsity of a list of tensors or a tensor given a threshold.
    Sparsity is the ratio of 0s to total elements.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    sparse_parameters = []
    if isinstance(parameters, torch.nn.Module):
        module = parameters
        sparse_parameters = []
        parameters = []
        for name, p in module.named_parameters():
            if '_signs' in name or '_negative' in name:
                continue
            elif check_any_parameter_in_network(module, name + '_signs'):
                sparse_parameters.append(p)
            else:
                parameters.append(p)

    total = 0
    zeros = 0
    for p in parameters:
        if threshold:
            zeros += (p.abs() < threshold).float().sum()
        else:
            zeros += (p == 0).float().sum()
        total += p.numel()
    
    for p in sparse_parameters:
        zeros += (p <= 0).float().sum()
        total += p.numel()
        
    return float(zeros/total)
