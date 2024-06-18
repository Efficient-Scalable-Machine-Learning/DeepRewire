import torch
import copy
from torch import nn
from src import DEEPR, SoftDEEPR, convert_to_deep_rewireable, convert_from_deep_rewireable
from src.utils import measure_sparsity
import matplotlib.pyplot as plt

"""
This is just an example to compare the different ways to handle bias in (soft)DEEPR.
We fit a fixed input tensor X to a fixed target tensor y over 100 iterations, we plot the loss as well as the inital and final sparsity of the model.
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
    model3 = copy.deepcopy(model)

    threshold = 1e-3
    init_sparsity = measure_sparsity(model.parameters(), threshold=threshold)
    convert_to_deep_rewireable(model, handle_biases='ignore')
    convert_to_deep_rewireable(model2, handle_biases='as_connections')
    convert_to_deep_rewireable(model3, handle_biases='second_bias')


    optimizer = SoftDEEPR(model.parameters(), lr=0.05, l1=0.005)
    optimizer2 = SoftDEEPR(model2.parameters(), lr=0.05, l1=0.005)
    optimizer3 = SoftDEEPR(model3.parameters(), lr=0.05, l1=0.005)
    criterium = nn.MSELoss()

    # data Tensor X and target y
    X = torch.rand(100, 500)
    y = torch.rand(100, 1)

    losses = []
    losses2 = []
    losses3 = []

    for epoch in range(100):

        # ignore
        pred = model(X)
        loss = criterium(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        pred2 = model2(X)
        loss2 = criterium(pred2, y)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        losses2.append(loss2.item())

        pred3 = model3(X)
        loss3 = criterium(pred3, y)
        optimizer3.zero_grad()
        loss3.backward()
        optimizer3.step()
        losses3.append(loss3.item())


    convert_from_deep_rewireable(model)
    convert_from_deep_rewireable(model2)
    convert_from_deep_rewireable(model3)

    final_sparsity = measure_sparsity(model.parameters())
    final_sparsity2 = measure_sparsity(model2.parameters())
    final_sparsity3 = measure_sparsity(model2.parameters())

    pred = model(X)
    loss = criterium(pred, y).detach()

    pred2 = model2(X)
    loss2 = criterium(pred2, y).detach()

    pred3 = model3(X)
    loss3 = criterium(pred3, y).detach()


    plt.plot(losses)
    plt.plot(losses2)
    plt.plot(losses3)
    plt.plot([loss for l in range(len(losses))], 'r--', linewidth=3)
    plt.plot([loss2 for l in range(len(losses))], 'g-', linewidth=2)
    plt.plot([loss3 for l in range(len(losses))], 'b-.', linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("MSE loss")
    plt.legend(["ignore", "as connections", "second bias", "ignore after reconversion", "as connections after reconversion", "second bias after reconversion"])
    plt.title(f"Initial sparsity (threshold {threshold}): {init_sparsity:.2f}\n"+
              f"Final sparsity 'ignore' (real zeros): {final_sparsity:.2f}\n"+
              f"Final sparsity 'as connections' (real zeros): {final_sparsity2:.2f}\n"+
              f"Final sparsity 'second bias' (real zeros): {final_sparsity3:.2f}\n")
    plt.show()
