"""
The optimizers for the rewiring network
"""

import torch
from torch.optim.optimizer import Optimizer, required

class DEEPR(Optimizer):
    """
    Deep-rewiring oftimizer with hard constraint on number of connections
    """
    def __init__(self, params, nc=required, lr=required, l1=0.0, reset_val=0.0, temp=None):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if nc is not required and nc < 0.0 or not isinstance(nc, int):
            raise ValueError(f"Invalid number of connections: {nc}")
        if l1 < 0.0:
            raise ValueError(f"Invalid L1 regularization term: {l1}")
        if reset_val < 0.0:
            raise ValueError(f"Invalid reset value: {reset_val}")

        if temp is None:
            temp = lr * l1**2 / 18

        self.nc = nc

        defaults = dict(lr=lr, l1=l1, temp=temp, reset_val=reset_val)
        super(DEEPR, self).__init__(params, defaults)

        # count parameters
        self.n_parameters = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad is False:
                    continue
                self.n_parameters += p.numel()

        if self.nc > self.n_parameters:
            raise ValueError("Number of connections can't be bigger than number"+
                            f"of parameters: nc:{nc} np:{self.n_parameters}")
        n_inactive = self.n_parameters - self.nc
        # set all indicies to active (positive), then deactivate randomly selected indicies.
        deactivate_indicies = torch.randperm(self.n_parameters)[:n_inactive]
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad is False:
                    continue
                p.data = torch.abs(p.data)
                org_size = p.data.size()
                d_idx = deactivate_indicies[deactivate_indicies < p.data.numel()]
                p.data = p.data.view(-1)
                p.data[d_idx] *= -1
                p.data = p.data.view(org_size)
                deactivate_indicies -= p.data.numel()
                deactivate_indicies = deactivate_indicies[deactivate_indicies >= 0]

    def attempt_activation(self, candidate_idx):
        """
        function will activate a connection if previously inactive and return 1.
        If was already active return 0.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if candidate_idx < p.data.numel():
                    org_size = p.data.size()
                    p.data = p.data.view(-1)
                    if p.data[candidate_idx] < 0:
                        p.data[candidate_idx] = group['reset_val']
                        p.data = p.data.view(org_size)
                        return 1
                    p.data = p.data.view(org_size)
                    return 0
                candidate_idx -= p.data.numel()
        return 0


    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # update step, then count of active connections
        active_connections = 0
        idx_counter = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # update step for active connections
                grad = p.grad.data
                noise = (2*group['lr']*group['temp'])**0.5 * torch.randn_like(p.data)
                mask = p.data >= 0
                p.data[mask] += -group['lr'] * (grad[mask] + group['l1']) + noise[mask]

                # count all active connections
                active_connections += torch.sum(p.data > 0).item()
                idx_counter += p.data.numel()

        # look how many connections are inactive and activate if necessary.
        # This is done by randomly sampling any index and trying.
        # If we assume a sparse network this should work better than
        # saving the active or inactive indicies beforehand.
        diff = self.nc - active_connections
        while diff > 0:
            candidate_index = torch.randint(low=0, high=self.n_parameters, size=(1,)).item()
            diff -= self.attempt_activation(candidate_index)

        return loss


class SoftDEEPR(Optimizer):
    """
    Deep-rewiring oftimizer with soft constraint on number of connections
    """
    def __init__(self, params, lr=required, l1=0.0, temp=None, min_weight=None):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if l1 < 0.0:
            raise ValueError(f"Invalid L1 regularization term: {l1}")
        if temp is None:
            temp = lr * l1**2 / 18
        if min_weight is None:
            min_weight = -3*lr

        defaults = dict(lr=lr, l1=l1, temp=temp, min_weight=min_weight)
        super(SoftDEEPR, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                noise = (2*group['lr']*group['temp'])**0.5 * torch.randn_like(p.data)

                mask = p.data >= 0

                p.data[mask] += -group['lr'] * (grad[mask] + group['l1']) + noise[mask]
                p.data[~mask] += noise[~mask].clamp(min=group['min_weight'])

        return loss

if __name__ == '__main__':
    pass
