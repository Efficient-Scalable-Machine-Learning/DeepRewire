import torch
import copy
from torch import nn
from src import SoftDEEPR, convert_to_deep_rewireable, convert_from_deep_rewireable
from src.utils import measure_sparsity, ProgressBar

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

"""
In this experiment, we will try to use a simple FCN and try it on MNIST instead of a fixed vector.
We let it run for 100 epochs.
"""

class someFCN(nn.Module):
    def __init__(self):
        super(someFCN, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return self.linear3(x)

def plot(losses, losses2, accuracies, accuracies2, final_loss=None, final_accuracy=None, init_sparsity=None, final_sparsity=None, final_sparsity2=None):
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))

    line1, = ax[0].plot(losses)
    line2, = ax[0].plot(losses2)
    if final_loss is not None:
        line3, = ax[0].plot([final_loss for _ in range(len(losses))], 'r--')
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("MSE loss (training)")

    ax[1].plot(accuracies)
    ax[1].plot(accuracies2)
    if final_accuracy is not None:
        ax[1].plot([final_accuracy for _ in range(len(accuracies))], 'r--')
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy (validation)")

    if final_loss is not None:
        lines = [line1, line2, line3]
        fig.legend(lines, ["SoftDEEPR", "SGD", "test of SoftDEEPR\nafter converting back"], loc='center right')
    else:
        lines = [line1, line2]
        fig.legend(lines, ["SoftDEEPR", "SGD"], loc='center right')

    if final_sparsity is not None:
        fig.suptitle(f"Initial sparsity (threshold {threshold}): {init_sparsity:.2f}\n" +
                     f"Final sparsity SoftDEEPR (real zeros): {final_sparsity:.2f}\n" +
                     f"Final sparsity SGD (threshold {threshold}): {final_sparsity2:.2f}\n")
    elif init_sparsity is not None:
        fig.suptitle(f"Initial sparsity (threshold {threshold}): {init_sparsity:.2f}")

    plt.subplots_adjust(right=0.95)
    plt.savefig('example3.png', bbox_inches='tight', dpi=200)
    plt.close()

if __name__ == '__main__':
    # Check if CUDA is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST data
    transform = transforms.ToTensor()
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    val_set_size = 200
    val_set, train_set = torch.utils.data.random_split(training_data, [val_set_size, len(training_data) - val_set_size])
    batch_size = 64
    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, batch_size=val_set_size)

    # Initialize models and move them to the device
    model = someFCN().to(device)
    model2 = copy.deepcopy(model).to(device)

    threshold = 1e-3
    init_sparsity = measure_sparsity(model.parameters(), threshold=threshold)
    convert_to_deep_rewireable(model, handle_biases='second_bias')

    eta = 0.05
    alpha = 1e-5
    T = eta * alpha**2 / 18
    optimizer = SoftDEEPR(model.parameters(), lr=eta, l1=alpha, temp=T)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=eta)
    criterium = nn.CrossEntropyLoss()

    losses = []
    losses2 = []
    accuracies = []
    accuracies2 = []

    pb = ProgressBar(max_value=100, length=30)

    for epoch in range(100):
        pb.prnt()
        pb.increment()

        # Validate models
        model.eval()
        model2.eval()
        with torch.no_grad():
            for Xv, yv in val_dataloader:
                Xv, yv = Xv.to(device), yv.to(device)
                pred = model(Xv)
                accuracies.append((pred.argmax(dim=1) == yv).float().mean().item())

                pred2 = model2(Xv)
                accuracies2.append((pred2.argmax(dim=1) == yv).float().mean().item())

        model.train()
        model2.train()

        avg_loss = 0
        avg_loss2 = 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            # SoftDEEPR
            pred = model(X)
            loss = criterium(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            # SGD
            pred2 = model2(X)
            loss2 = criterium(pred2, y)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            avg_loss2 += loss2.item()

        losses.append(avg_loss / len(train_dataloader))
        losses2.append(avg_loss2 / len(train_dataloader))
        plot(losses, losses2, accuracies, accuracies2, init_sparsity=init_sparsity)

    # Convert back to standard FCN and measure final sparsity
    convert_from_deep_rewireable(model)
    final_sparsity = measure_sparsity(model.parameters())
    final_sparsity2 = measure_sparsity(model2.parameters(), threshold=threshold)

    # Check if still working after reconversion
    with torch.no_grad():
        for Xv, yv in val_dataloader:
            Xv, yv = Xv.to(device), yv.to(device)
            pred = model(Xv)
            accuracy = (pred.argmax(dim=1) == yv).float().mean().item()

    plot(losses, losses2, accuracies, accuracies2, final_loss=loss.item(), final_accuracy=accuracy, init_sparsity=init_sparsity, final_sparsity=final_sparsity, final_sparsity2=final_sparsity2)
