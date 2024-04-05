import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import itertools
import wandb

class Network(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, dense_nodes, conv_kernel_size, pool_kernel_size, num_epochs):
        super().__init__()
        self.customNN = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=conv_kernel_size),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
            nn.ReLU(),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=conv_kernel_size),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
            nn.ReLU(),
            nn.Flatten()
        )
        # Calculate the size of the flattened feature map after the conv layers
        # Assumes input image size is 28x28
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            features = self.customNN(dummy_input)
            self.num_flattened_features = features.size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_flattened_features, dense_nodes),
            nn.ReLU(),
            nn.Linear(dense_nodes, 10)
        )

    def forward(self, x):
        x = self.customNN(x)
        x = self.classifier(x)
        return x


def train_model(train_loader, model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        wandb.log({"epoch_loss": epoch_loss, "epoch_accuracy": epoch_accuracy})
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {100 * epoch_accuracy:.2f}%")

def test_model(test_loader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()

    accuracy = correct / total
    test_loss /= len(test_loader.dataset)
    wandb.log({"test_loss": test_loss, "test_accuracy": accuracy})
    print(f"Test Accuracy: {100 * accuracy:.2f}%, Test Loss: {test_loss:.4f}")
    return accuracy, test_loss

def main(argv):
    # Hyperparameters
    conv1_filters = [128, 150]
    conv2_filters = [320, 128]
    dense_nodes = [512, 128]
    conv_kernel_sizes = [4, 2]
    pool_kernel_sizes = [3, 4]
    num_epochs = [8,12]
    
    # Data loading and preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize WandB
    wandb.init(project="final_MNIST_Fassion")
    best_accuracy = 0.0
    best_hyperparameters = None
    
    # linear search order
    parameters = [conv1_filters, conv2_filters, dense_nodes, conv_kernel_sizes, pool_kernel_sizes, num_epochs]
    parameter_names = ['conv1_filters', 'conv2_filters', 'dense_nodes', 'conv_kernel_size', 'pool_kernel_size', 'num_epochs']
    
    for i in range(len(parameters)):
        for value in parameters[i]:
            hyperparameters = {}
            for j in range(len(parameters)):
                if i == j:
                    hyperparameters[parameter_names[j]] = value
                else:
                    hyperparameters[parameter_names[j]] = parameters[j][0]
    
            # initialize model with current hyperparameters
            model = Network(**hyperparameters).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())
    
            # start WandB run
            run_name = '_'.join([f"{param}_{val}" for param, val in hyperparameters.items()])
            run = wandb.init(project="final_MNIST_Fassion", config=hyperparameters, name=run_name)
    
            print(f"\nTesting with hyperparameters: {hyperparameters}")
            train_model(train_loader, model, criterion, optimizer, hyperparameters['num_epochs'])
            accuracy, _ = test_model(test_loader, model, criterion)
    
            # checking if current model has the best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparameters = hyperparameters
            # finishing WandB run
            run.finish()
    
    print("\nBest hyperparameters:", best_hyperparameters)
    print("Best accuracy:", best_accuracy)