import torch
import copy
from torch import nn
from deep_rewire import SoftDEEPR, convert, reconvert
from deep_rewire.utils import measure_sparsity
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FCNN()
    model.to(device)

    sparse_params, _ = convert(model, active_probability=0.05)
    criterion = nn.CrossEntropyLoss()
    optimizer = SoftDEEPR(sparse_params, lr=0.05, l1=1e-5)

    num_epochs = 200
    train_losses = []
    val_accuracies = []
    sparsities = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        with torch.no_grad():
            sparsities.append(measure_sparsity(model)*100)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.2f}%, Sparsity: {sparsities[-1]:.2f}%')

    reconvert(model)
    sparsity = measure_sparsity(model.parameters())
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(train_losses, label='Training Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy (%)', color='tab:orange')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax3 = ax1.twinx()
    ax3.set_ylabel('Sparsity (%)', color='tab:green')
    ax3.plot(sparsities, label='Sparsity', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    ax3.spines['right'].set_position(('outward', 60))

    fig.tight_layout()
    plt.title(f'SoftDEEPR on MNIST')

    torch.save(model.state_dict(), 'sparse_model.pth')
    plt.savefig("mnist_softdeepr.svg", format='svg', bbox_inches='tight')
