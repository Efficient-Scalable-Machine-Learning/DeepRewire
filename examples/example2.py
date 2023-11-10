import torch
import copy
from torch import nn
from src import softDEEPR, convert_to_deep_rewireable, convert_from_deep_rewireable
from src.utils import measure_sparsity

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

"""
In this experiment we will try to use a simple FCN and try it on MNIST instead of a fixed vector.
We let it run only for one epoch (one time over training data) but plot after each batch.
"""

class someFCN(nn.Module):

	def __init__(self):
		super(someFCN, self).__init__()
		self.linear1 = nn.Linear(28*28, 512)
		self.linear2 = nn.Linear(512, 256)
		self.linear3 = nn.Linear(256, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = x.view(-1, 28*28)
		x = self.linear1(x)
		x = self.relu(x)
		x = self.linear2(x)
		x = self.relu(x)
		return self.linear3(x)


if __name__ == '__main__':

	training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
	)
	
	test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
	)

	val_set_size = 200
	val_set, train_set = torch.utils.data.random_split(training_data, [val_set_size, len(training_data)-val_set_size])
	batch_size = 64
	train_dataloader = DataLoader(train_set, batch_size=batch_size)
	val_dataloader = DataLoader(val_set, batch_size=val_set_size)

	model = someFCN()
	model2 = copy.deepcopy(model)

	threshold = 1e-3
	init_sparsity = measure_sparsity(model.parameters(), threshold=threshold)
	convert_to_deep_rewireable(model)

	eta = 0.05
	alpha = 1e-5
	T = eta*alpha**2/18
	optimizer = softDEEPR(model.parameters(), lr=eta, l1=alpha, temp=T)
	optimizer2 = torch.optim.SGD(model2.parameters(), lr=eta)
	criterium = nn.CrossEntropyLoss()

	losses = []
	losses2 = []

	accuracies = []
	accuracies2 = []

	for batch, (X, y) in enumerate(train_dataloader):
		
		if batch % 10 == 0:
			for _, (Xv, yv) in enumerate(val_dataloader):

				with torch.no_grad():
					# softDEEPR
					pred = model(Xv)
					accuracies.append((pred.argmax(dim=1) == yv).float().mean().item())
		
					# SGD
					pred2 = model2(Xv)
					accuracies2.append((pred2.argmax(dim=1) == yv).float().mean().item())
				

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
	for _, (Xv, yv) in enumerate(val_dataloader):
		with torch.no_grad():
			pred = model(Xv)
			accuracy = (pred.argmax(dim=1) == yv).float().mean().item()
		

	fig, ax = plt.subplots(1, 2)
			
	line1, = ax[0].plot(losses)
	line2, = ax[0].plot(losses2)
	line3, = ax[0].plot([loss for _ in range(len(losses))], 'r--')
	ax[0].set_xlabel("batch")
	ax[0].set_ylabel("MSE loss (training)")

	ax[1].plot(accuracies)
	ax[1].plot(accuracies2)
	ax[1].plot([accuracy for _ in range(len(accuracies))], 'r--')
	ax[1].set_xlabel("batch")
	ax[1].set_ylabel("accuracy (validation)")
	
	lines = [line1, line2, line3]
	fig.legend(lines, ["softDEEPR", "SGD", "test of softDEEPR\nafter converting back"], loc='center right')
	fig.suptitle(f"Initial sparsity (threshold {threshold}): {init_sparsity:.2f}\n"+
			  f"Final sparsity softDEEPR (real zeros): {final_sparsity:.2f}\n"+
			  f"Final sparsity SGD (threshold {threshold}): {final_sparsity2:.2f}\n")
	plt.subplots_adjust(right=0.85)
	plt.show()

