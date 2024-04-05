# Aaron Pan/Abhishek Uddaraju
# Task 2: Loading first layers of network from task 1

# import statements
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import ToTensor
import numpy as np
import cv2

# Contains Task 2 and Extension for visualizing model weights and filtered images
def main(argv):
    # handle any command line arguments in argv
    # loading model
    #load model
    model = torch.load('data/weights/CustomNetwork_MNIST.pth', map_location=torch.device('cpu'))
    #print(model)

    #extract first layer
    conv1_weights = model['customNN.0.weight']
    N = conv1_weights.size(0)

    #display filter outputs
    fig, axes = plt.subplots(N//4, 4, figsize=(10, N))
    for i, ax in enumerate(axes.flat):
        ax.imshow(conv1_weights[i, 0])
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show(block = False)

    #training/test datasets and data loaders
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #getting image form dataset
    image, label = next(iter(test_loader))
    image_np = image.squeeze().numpy()

    # Apply  All filters
    N = conv1_weights.size(0)
    filtered_images = []

    with torch.no_grad():
        for i in range(N):
            filter_result = cv2.filter2D(image_np, -1, conv1_weights[i, 0].cpu().numpy())
            filtered_images.append(filter_result)

    # Plotting filters and outputs
    fig, axes = plt.subplots(N, 2, figsize=(6, N*2))
    for i in range(N):
        axes[i,0].imshow(conv1_weights[i, 0].cpu(), cmap='gray')
        axes[i,0].set_title(f'Filter {i+1}')
        axes[i,0].axis('off')
    for i in range(N):
        axes[i,1].imshow(filtered_images[i], cmap='gray')
        axes[i,1].set_title(f'Output {i+1}')
        axes[i,1].axis('off')
    plt.tight_layout()
    plt.show(block = False)

    # Initialize model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    conv1_weights = model.conv1.weight.data

    # Plotting first weights
    N = conv1_weights.size(0)
    fig, axes = plt.subplots(N // 8 + 1, 8, figsize=(20, 2.5 * (N // 8 + 1)))
    for i, ax in enumerate(axes.flat):
        if i < N:
            ax.imshow(conv1_weights[i, 0].cpu().numpy())
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show(block = False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the input size of ResNet50
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channel
        transforms.ToTensor(),           # Convert images to PyTorch tensors
    ])

    #get mnist dataset from pytorch
    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)

    #change model to eval mode
    model.eval()

    #list to keep images for plotting
    filtered_images = []

    #processing thru filter
    with torch.no_grad():
        for images, labels in mnist_loader:
            # Feed through the first layer
            conv1_output = model.conv1(images)

            num_filters = conv1_output.size(1)

            for i in range(num_filters):
                filter_output = conv1_output[0, i].cpu().numpy()
                filtered_images.append(filter_output)
            break

    num_rows = min(N, 8)
    num_cols = (N // num_rows) + (N % num_rows > 0)

    # Plot images
    fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(4 * num_cols, 2 * num_rows))
    for i in range(N):
        row = i % num_rows
        col = i // num_rows

        axes[row, 2 * col].imshow(conv1_weights[i, 0].cpu(), cmap='gray')
        axes[row, 2 * col].set_title(f'Filter {i + 1}')
        axes[row, 2 * col].axis('off')

        axes[row, 2 * col + 1].imshow(filtered_images[i], cmap='gray')
        axes[row, 2 * col + 1].set_title(f'Output {i + 1}')
        axes[row, 2 * col + 1].axis('off')

    plt.tight_layout()
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)