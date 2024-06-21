"""
The optimizers for the rewiring network
"""

import torch
from torch.optim.optimizer import Optimizer, required

class DEEPR(Optimizer):
    """
    Deep-rewiring oftimizer with hard constraint on number of connections
    """
    def __init__(self, params, nc=required, lr=0.05, l1=1e-4, reset_val=0.0, temp=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if nc is not required and nc < 0.0 or not isinstance(nc, int):
            raise ValueError(f"Invalid number of connections: {nc}")
        if l1 < 0.0:
            raise ValueError(f"Invalid L1 regularization term: {l1}")
        if reset_val < 0.0:
            raise ValueError(f"Invalid reset value: {reset_val}")

        if temp is None:
            temp = lr / 210

        self.nc = nc

        defaults = dict(lr=lr, l1=l1, temp=temp, reset_val=reset_val)
        super(DEEPR, self).__init__(params, defaults)

        # count parameters
        self.n_parameters = 0
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                self.n_parameters += p.numel()

        if self.nc > self.n_parameters:
            raise ValueError("Number of connections can't be bigger than number"+
                            f"of parameters: nc:{nc} np:{self.n_parameters}")

        activate_indices = self.sample_unique_indices(self.nc, self.n_parameters)
        self.init_activation(activate_indices)


    def sample_unique_indices(self, length, max_int):
        if length > max_int:
            raise ValueError("Cannot sample more unique indices than the size of the range.")
        
        selected_indices = set()
        
        while len(selected_indices) < length:
            new_indices = torch.randint(0, max_int, (length - len(selected_indices),))
            selected_indices.update(new_indices.tolist())

        return torch.tensor(list(selected_indices))


    def init_activation(self, activate_indices):
        """
        Function to initialize activation by flipping the sign of the selected indices.
        """
        remaining_indices = activate_indices

        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                num_elements = p.data.numel()

                p.data = -torch.abs(p.data)
                
                p_data_flat = p.data.view(-1)
                
                in_current_param_mask = remaining_indices < num_elements
                current_indices = remaining_indices[in_current_param_mask]
                
                if current_indices.numel() > 0:
                    p_data_flat[current_indices] *= -1
                    p_data_flat[current_indices] = torch.clamp(p_data_flat[current_indices], min=group['reset_val'])
                    
                    remaining_indices = remaining_indices[~in_current_param_mask]
                    
                remaining_indices -= num_elements
                
                if remaining_indices.numel() == 0:
                    break

    def attempt_activation(self, candidate_indices):
        """
        Function will activate connections if previously inactive and return the number of activations.
        """
        activations = 0
        remaining_indices = candidate_indices
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                p_data_flat = p.data.view(-1)
                num_elements = p_data_flat.numel()
                
                in_current_param_mask = remaining_indices < num_elements
                current_indices = remaining_indices[in_current_param_mask]
                
                if current_indices.numel() > 0:
                    selected_values = p_data_flat[current_indices]
                    
                    to_activate_mask = selected_values < 0
                    to_activate_indices = current_indices[to_activate_mask]
                    
                    if to_activate_indices.numel() > 0:
                        p_data_flat[to_activate_indices] = group['reset_val']
                        activations += to_activate_indices.numel()
                
                remaining_indices = remaining_indices[~in_current_param_mask] - num_elements
                
                if remaining_indices.numel() == 0:
                    break
        
        return activations

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active_connections = 0
        idx_counter = 0
        for group in self.param_groups:
            lr = group['lr']
            l1 = group['l1']
            temp = group['temp']
            sqrt_temp = (2 * lr * temp) ** 0.5
           
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                noise = sqrt_temp * torch.randn_like(p.data)
                mask = p.data >= 0
                p.data += mask.float() * (-lr * (grad + l1) + noise)

                active_connections += torch.sum(p.data > 0).item()
                idx_counter += p.data.numel()

        # look how many connections are inactive and activate if necessary.
        diff = self.nc - active_connections
        while diff > 0:
            candidate_indices = torch.randint(low=0, high=self.n_parameters, size=(diff,))
            candidate_indices = candidate_indices.to(self.param_groups[0]['params'][0].device)
            diff -= self.attempt_activation(candidate_indices)

        return loss


class SoftDEEPR(Optimizer):
    """
    Deep-rewiring oftimizer with soft constraint on number of connections
    """
    def __init__(self, params, lr=0.05, l1=1e-5, temp=None, min_weight=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if l1 < 0.0:
            raise ValueError(f"Invalid L1 regularization term: {l1}")
        if temp is None:
            temp = lr * l1**2 / 18
        if min_weight is None:
            min_weight = -3*l1

        defaults = dict(lr=lr, l1=l1, temp=temp, min_weight=min_weight)
        super(SoftDEEPR, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            l1 = group['l1']
            temp = group['temp']
            min_weight = group['min_weight']
            sqrt_temp = (2 * lr * temp) ** 0.5
           
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                noise = sqrt_temp * torch.randn_like(p.data)

                mask = p.data >= 0
               
                """
                p.data += mask.float() * (-lr * (grad + l1) + noise)
                p.data += (~mask).float() * noise.clamp(min=min_weight)
                """

                # this is how its done in the paper i think:
                p.data += noise - mask.float() * lr * (grad + l1)
                p.data = p.data.clamp(min=min_weight)
        return loss

if __name__ == '__main__':
    pass
