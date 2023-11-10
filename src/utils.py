import torch
from torch import nn
import time    

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

class progressBar:

	def __init__(self, max_value: int, length: int):
		self.counter = 0
		self.length = length
		self.max_value = max_value

	def increment(self, step: int = 1):
		self.counter += step
		if self.counter > self.max_value:
			self.counter = self.max_value

	def prnt(self):
		return print(' Progress: ['+('-'*int(self.counter/self.length)).ljust(self.length, ' ')+']', end = '\r')

	def reset(self):
		self.counter = 0

if __name__ == '__main__':
	a = progressBar(100, 20)
	while True:
		a.prnt()
		a.increment()
		time.sleep(0.2)
