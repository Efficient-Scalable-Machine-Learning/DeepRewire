import torch
from torch.optim.optimizer import Optimizer, required

class softDEEPR(Optimizer):
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
		super(softDEEPR, self).__init__(params, defaults)

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

				# Apply a custom update based on the sign of the parameter values
				mask = p.data >= 0

				# Update rule for non-negative parameter values
				p.data[mask] += -group['lr'] * (grad[mask] + group['l1']) + noise[mask]
				# Update rule for negative parameter values
				p.data[~mask] += noise[~mask].clamp(min=group['min_weight'])

		return loss

if __name__ == '__main__':
	pass
