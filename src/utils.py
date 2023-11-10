import torch
from torch import nn

def measure_sparsity(parameters, threshold=0):
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	total = 0
	zeros = 0
	for p in parameters:
		if threshold:
			zeros += (p.abs() < threshold).float().sum()
		else:
			zeros += (p == 0).float().sum()
		total += p.numel()
	return float(zeros/total)
