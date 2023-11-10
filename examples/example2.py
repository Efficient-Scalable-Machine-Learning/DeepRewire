import torch
import copy
from torch import nn
from src import softDEEPR, convert_to_deep_rewireable, convert_from_deep_rewireable
from src.utils import measure_sparsity
import matplotlib.pyplot as plt

"""
In this experiment we will try to use a simple FCN and try it on MNIST instead of a fixed vector
"""

class someFCN(nn.Module):

	def __init__(self):
		super(someFCN, self).__init__()
		self.linear1 = nn.Linear(500, 300)
		self.linear2 = nn.Linear(300, 100)
		self.linear3 = nn.Linear(100, 50)
		self.linear4 = nn.Linear(50, 1)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.linear1(x)
		x = self.relu(x)
		x = self.linear2(x)
		x = self.relu(x)
		x = self.linear3(x)
		x = self.relu(x)
		return self.linear4(x)


if __name__ == '__main__':
	model = someFCN()
	model2 = copy.deepcopy(model)

	threshold = 1e-3
	init_sparsity = measure_sparsity(model.parameters(), threshold=threshold)
	convert_to_deep_rewireable(model)
	optimizer = softDEEPR(model.parameters(), lr=0.05, l1=0.005)
	optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.05)
	criterium = nn.MSELoss()

	# data Tensor X and target y
	X = torch.rand(100, 500)
	y = torch.rand(100, 1)

	losses = []
	losses2 = []

	for epoch in range(100):

		# softDEEPR
		pred = model(X)
		loss = criterium(pred, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(loss.item())

		# SGD
		pred2 = model2(X)
		loss2 = criterium(pred2, y)
		optimizer2.zero_grad()
		loss2.backward()
		optimizer2.step()
		losses2.append(loss2.item())

	convert_from_deep_rewireable(model)
	final_sparsity = measure_sparsity(model.parameters())
	final_sparsity2 = measure_sparsity(model2.parameters(), threshold=threshold)

	pred = model(X)
	loss = criterium(pred, y).detach()
	plt.plot(losses)
	plt.plot(losses2)
	plt.plot([loss for l in range(len(losses))], 'r--')
	plt.xlabel("epoch")
	plt.ylabel("MSE loss")
	plt.legend(["softDEEPR", "SGD", "test of softDEEPR after converting back"])
	plt.title(f"Initial sparsity (threshold {threshold}): {init_sparsity:.2f}\n"+
			  f"Final sparsity softDEEPR (real zeros): {final_sparsity:.2f}\n"+
			  f"Final sparsity SGD (threshold {threshold}): {final_sparsity2:.2f}\n")
	plt.show()
