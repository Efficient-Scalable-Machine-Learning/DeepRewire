import torch
import copy
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
from src import SoftDEEPR, DEEPR, convert_to_deep_rewireable, convert_from_deep_rewireable
from src.utils import measure_sparsity

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

def plot_results(optimizer_results, init_sparsity):
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    for name, results in optimizer_results.items():
        losses, accuracies, final_loss, final_accuracy, final_sparsity, after_conversion_accuracy = results
        ax[0].plot(losses, label=f'{name} Training Loss')
        ax[1].plot(accuracies, label=f'{name} Validation Accuracy')
        if final_loss is not None:
            ax[0].plot([final_loss for _ in range(len(losses))], 'r--', label=f'{name} Final Loss')
        if final_accuracy is not None:
            ax[1].plot([final_accuracy for _ in range(len(accuracies))], 'r--', label=f'{name} Final Accuracy')
        if after_conversion_accuracy is not None:
            ax[1].plot([after_conversion_accuracy for _ in range(len(accuracies))], 'g--', label=f'{name} After Conversion Accuracy')

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Training Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Validation Accuracy")
    ax[0].legend()
    ax[1].legend()

    fig.suptitle(f"Initial sparsity: {init_sparsity:.2f}")
    plt.subplots_adjust(right=0.95)
    plt.savefig('example3.png', bbox_inches='tight', dpi=200)
    plt.close()

def train_validate_model(model, optimizer, train_loader, val_loader, criterium, device, epochs, is_deep_rewireable=False, optimizer_name=""):
    losses, accuracies = [], []
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterium(pred, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(train_loader)
        losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            accuracy = 0
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                pred = model(Xv)
                accuracy += (pred.argmax(dim=1) == yv).float().mean().item()
            accuracy /= len(val_loader)
            accuracies.append(accuracy)

        # Log training and validation metrics to wandb
        wandb.log({f"{optimizer_name} Training Loss": avg_loss, f"{optimizer_name} Validation Accuracy": accuracy, "epoch": epoch})

    return losses, accuracies

if __name__ == '__main__':
    wandb.init(project="mnist-optimization-comparison")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    val_set_size = 200
    val_set, train_set = torch.utils.data.random_split(training_data, [val_set_size, len(training_data) - val_set_size])
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=val_set_size)

    model = someFCN()
    model2 = copy.deepcopy(model)
    model3 = copy.deepcopy(model)
    param_total = sum(p.numel() for p in model.parameters())
    nc = int(param_total * 0.3)
 
    threshold = 1e-3
    init_sparsity = measure_sparsity(model.parameters(), threshold=threshold)
    convert_to_deep_rewireable(model, handle_biases='second_bias')
    convert_to_deep_rewireable(model3, handle_biases='second_bias')

    model.to(device)
    model2.to(device)
    model3.to(device)

    eta = 0.05
    alpha = 1e-5
    T = eta * alpha**2 / 18
    optimizer = SoftDEEPR(model.parameters(), lr=eta, l1=alpha, temp=T)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=eta)
    optimizer3 = DEEPR(model3.parameters(), nc=nc, lr=eta, l1=alpha, temp=T)
    criterium = nn.CrossEntropyLoss()

    optimizer_results = {}

    losses, accuracies = train_validate_model(model, optimizer, train_loader, val_loader, criterium, device, 100, is_deep_rewireable=True, optimizer_name="SoftDEEPR")
    convert_from_deep_rewireable(model)
    final_sparsity = measure_sparsity(model.parameters())
    final_accuracy = None
    with torch.no_grad():
        final_accuracy = 0
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(device), yv.to(device)
            pred = model(Xv)
            final_accuracy += (pred.argmax(dim=1) == yv).float().mean().item()
        final_accuracy /= len(val_loader)

    optimizer_results['SoftDEEPR'] = (losses, accuracies, losses[-1], accuracies[-1], final_sparsity, final_accuracy)

    losses2, accuracies2 = train_validate_model(model2, optimizer2, train_loader, val_loader, criterium, device, 100, optimizer_name="SGD")
    final_sparsity2 = measure_sparsity(model2.parameters(), threshold=threshold)

    optimizer_results['SGD'] = (losses2, accuracies2, losses2[-1], accuracies2[-1], final_sparsity2, None)

    losses3, accuracies3 = train_validate_model(model3, optimizer3, train_loader, val_loader, criterium, device, 100, is_deep_rewireable=True, optimizer_name="DEEPR")
    convert_from_deep_rewireable(model3)
    final_sparsity3 = measure_sparsity(model3.parameters())
    final_accuracy3 = None
    with torch.no_grad():
        final_accuracy3 = 0
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(device), yv.to(device)
            pred = model3(Xv)
            final_accuracy3 += (pred.argmax(dim=1) == yv).float().mean().item()
        final_accuracy3 /= len(val_loader)

    optimizer_results['DEEPR'] = (losses3, accuracies3, losses3[-1], accuracies3[-1], final_sparsity3, final_accuracy3)

    plot_results(optimizer_results, init_sparsity)
    wandb.finish()
