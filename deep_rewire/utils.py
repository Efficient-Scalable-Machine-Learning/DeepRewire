"""
provides some useful functions
"""

import torch


def measure_sparsity(parameters, threshold=0):
    """
    Measures sparsity of a list of tensors or a tensor given a threshold.
    Sparsity is the ratio of 0s to total elements.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    rewireable = False
    if isinstance(parameters, torch.nn.Module):
        if any('_signs' in n for n, _ in parameters.named_parameters()):
            rewireable = True
        parameters = [p for n, p in parameters.named_parameters() if all(
            x not in n for x in ['_signs', '_negative'])]

    total = 0
    zeros = 0
    for p in parameters:
        if threshold:
            zeros += (p.abs() < threshold).float().sum()
        elif rewireable:
            zeros += (p <= 0).float().sum()
        else:
            zeros += (p == 0).float().sum()
        total += p.numel()
    return float(zeros/total)
