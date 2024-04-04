# Aaron Pan
# Training NN on MNIST data

# import statements
import sys
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.customNN = nn.Sequential(
          nn.Conv2d(1,10, kernel_size = 5),
          nn.MaxPool2d(kernel_size = 2),
          nn.ReLU(),
          nn.Conv2d(10,20, kernel_size = 5),
          nn.Dropout(p = 0.5),
          nn.MaxPool2d(kernel_size = 2),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(20*4*4, 50),
          nn.ReLU(),
          nn.Linear(50,10),
          nn.LogSoftmax()
        )

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = self.customNN(x)
        return x

# FUNCTIONS:

# Runs one epoch to train the model
def train_network(dataloader, model, loss_fn, optimizer, batch_size, acc_plt, loss_plt, sample_train):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # for accuracy in train
    correct_train = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct_train += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            accuracy_train = correct_train / current
            print(f"loss: {loss:>7f}  accuracy: {(100*accuracy_train):>0.1f}% [{current:>5d}/{size:>5d}]")
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            if (len(sample_train) == 0):
              sample_train.append(current)
            else:
              sample_train.append(sample_train[-1]+current)
            acc_plt.append(accuracy_train)
            loss_plt.append(loss)

# Runs one epoch to test the network
def test_network(dataloader, model, loss_fn, acc_plt, loss_plt, sample_train, sample_test):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    acc_plt.append(correct)
    loss_plt.append(test_loss)
    sample_test.append(sample_train[-1])

# MAIN FUNCTION

# Main loop to train, test, and save the network to a file
def main(argv):
    # handle any command line arguments in argv

    # main function code
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    batch_size = 32

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(0, cols * rows):
        # sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
        img, label = test_dataset[i]
        figure.add_subplot(rows, cols, i+1)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show(block=False)
    plt.pause(0.1)

    model = MyNetwork().to(device)
    print(model)

    learning_rate = 1e-2
    epochs = 12
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    train_samples = []
    test_samples = []


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_network(train_loader, model, loss_fn, optimizer, batch_size, train_acc, train_loss, train_samples)
        test_network(test_loader, model, loss_fn, test_acc, test_loss, train_samples, test_samples)
    print("Done!")

    file_name = 'data/weights/CustomNetwork_MNIST.pth'
    file_path = os.path.join(os.getcwd(), file_name)

    torch.save(model.state_dict(), file_path)

    #accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_samples, train_acc, label='Training Accuracy')
    plt.scatter(test_samples, test_acc, color='red', label='Test Accuracy')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=False)

    plt.figure(figsize=(10, 5))
    plt.plot(train_samples, train_loss, label='Training Accuracy')
    plt.scatter(test_samples, test_loss, color='red', label='Test Accuracy')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Negative Log Likelyhood Loss')
    plt.legend()
    plt.show()
    
    return

if __name__ == "__main__":
    main(sys.argv)