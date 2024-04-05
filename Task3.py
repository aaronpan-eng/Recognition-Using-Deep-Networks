# Aaron Pan/Abhishek Uddaraju
# Task 3: Greek letter dataset

# import statements
import sys
import torch
from torch import nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# class definitions
# Network class
class Network(nn.Module):
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
          nn.Linear(50,10)
        )

    def forward(self, x):
        x = self.customNN(x)
        return x

#greek transforms for imput data
class GreekTransform:
    def __init__(self):
        pass
    def __call__(self, x):
        x = torchvision.transforms.functional.resize(x, (133, 133))
        x = torchvision.transforms.functional.gaussian_blur(x, kernel_size=(3, 3), sigma=(0.01, 5))
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        noise = torch.randn(x.shape) * 0.01
        x = x + noise
        return torchvision.transforms.functional.invert( x )

# Function to plot images with labels
def plot_images_labels(images, labels, classes):
    fig, axes = plt.subplots(1,5, figsize=(15, 3))
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx].permute(1, 2, 0).squeeze(), cmap='gray')
        ax.set_title(classes[labels[idx]])
        ax.axis('off')
    plt.show()

# Function to plot images with labels
def plot_images_labels(images, labels, classes):
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx].permute(1, 2, 0).squeeze(), cmap='gray')
        ax.set_title(classes[labels[idx]])
        ax.axis('off')
    plt.show()

def train_loop(dataloader, model, loss_fn, optimizer, acc_plt, loss_plt, sample_train, device, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    # for accuracy in train
    correct_train = 0
    count = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct_train += (pred.argmax(1) == y).type(torch.float).sum().item()
        count += len(pred)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            accuracy_train = correct_train / count
            print(f"loss: {loss:>7f}  accuracy: {(100*accuracy_train):>0.1f}% [{current:>5d}/{size:>5d}]")
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            if (len(sample_train) == 0):
              sample_train.append(current)
            else:
              sample_train.append(sample_train[-1]+current)
            acc_plt.append(accuracy_train)
            loss_plt.append(loss)


def test_loop(dataloader, model, loss_fn, acc_plt, loss_plt, sample_train, sample_test, device):
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
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    acc_plt.append(correct)
    loss_plt.append(test_loss)
    sample_test.append(sample_train[-1])

#calcualte class accuracy
def class_wise_accuracy(model, test_loader, classes, device):
    class_correct = [0 for _ in range(3)]  # Assuming there are 3 classes
    class_total = [0 for _ in range(3)]

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c.item()
                class_total[label] += 1

    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(3)]
    return class_accuracy


# main function to handle plotting, training, testing on greek letters
def main(argv):
    # handle any command line arguments in argv
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = Network().to(device)
    print(model)

    model.load_state_dict(torch.load("data/weights/CustomNetwork_MNIST.pth"))

    # freezes the parameters for the whole network4
    for param in model.parameters():
        param.requires_grad = False

    model.customNN[10] = torch.nn.Linear(50,3)
    model.to(device)

    for param in model.customNN[10].parameters():
        param.requires_grad = True
    import torchsummary
    torchsummary.summary(model, (1,28,28))

    torch.cuda.is_available()

    # DataLoader for the Greek data set
    training_set_path = "data/greek_train"
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( training_set_path,
                                        transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                        GreekTransform(), torchvision.transforms.Normalize((0.1307,), (0.3081,) ) ] ) ),
                                            batch_size = 5,shuffle = True )

    # DataLoader for the Greek data set
    testing_set_path = "data/greek_test"
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(testing_set_path,
                                        transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                        GreekTransform(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                        batch_size = 1,shuffle = True)

    # Map class indices to class names
    classes = train_loader.dataset.classes

    # Iterate through the data loader and plot images with labels
    for images, labels in train_loader:
        plot_images_labels(images, labels, classes)
        break  # Only plot the first batch

    # Map class indices to class names
    classes = test_loader.dataset.classes

    # Iterate through the data loader and plot images with labels
    for images, labels in test_loader:
        print(labels)
        plot_images_labels(images, labels, classes)
        break  # Only plot the first batch

    learning_rate = 1e-3
    batch_size = 5
    epochs = 30

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #initilaizing lists for later use
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    train_samples = []
    test_samples = []


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, train_acc, train_loss, train_samples, device, batch_size)
        test_loop(test_loader, model, loss_fn, test_acc, test_loss, train_samples, test_samples, device)
    print("Done!")

    #accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_samples, train_acc, label='Training Accuracy')
    plt.scatter(test_samples, test_acc, color='red', label='Test Accuracy')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block = False)

    #loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_samples, train_loss, label='Training Loss')
    plt.scatter(test_samples, test_loss, color='red', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Negative Log Likelyhood Loss')
    plt.legend()
    plt.show(block = False)

    classes = test_loader.dataset.classes
    print(classes)
    # Example usage:
    class_acc = class_wise_accuracy(model, test_loader, classes, device)
    print("Class-wise Accuracy:")
    for i, acc in enumerate(class_acc):
        print(f"Class {classes[i]}: {acc:.2f}%")

    #saving model file with weights 
    torch.save(model.state_dict(), 'data/weights/task_3.pth')
    return

if __name__ == "__main__":
    main(sys.argv)