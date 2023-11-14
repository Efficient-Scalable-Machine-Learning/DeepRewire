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
    total = 0
    zeros = 0
    for p in parameters:
        if threshold:
            zeros += (p.abs() < threshold).float().sum()
        else:
            zeros += (p == 0).float().sum()
        total += p.numel()
    return float(zeros/total)

class ProgressBar:
    """
    Depicts progress
    """

    def __init__(self, max_value: int, length: int):
        self.counter = 0
        self.length = length
        self.max_value = max_value

    def increment(self, step: int = 1):
        """
        Increment the counter by step
        """
        self.counter += step
        if self.counter > self.max_value:
            self.counter = self.max_value

    def prnt(self):
        """
        Print the bar to console
        """
        print('  Progress: ['+('-'*int(self.counter*self.length/self.max_value)).ljust(self.length,
                            ' ')+']', end = '\r')

    def reset(self):
        """
        Reset the bar
        """
        self.counter = 0

if __name__ == '__main__':
    pass
