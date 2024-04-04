# Aaron Pan/Abhishek Uddaraju
# Testing the network on MNIST

# import statements
import sys
import cv2
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

# Custom network from task 1
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

# Tests network from task 1
def main(argv):
    model = MyNetwork()
    model.load_state_dict(torch.load('data/weights/CustomNetwork_MNIST.pth'))
    model.eval()

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    predictions = []
    images_to_plot = []

    with torch.no_grad():
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(test_loader):
            if i >= 10:
                break
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            images_to_plot.append(images.squeeze().numpy())

            # Print output values and index of max output value
            print(f'Example {i + 1}:')
            print('Output values:', ['%.2f' % val for val in outputs.squeeze().tolist()])
            print('Index of max output value:', predicted.item())
            print('Correct label:', labels.item())
            print()

        accuracy = correct / total
        print(f'Accuracy on 10 examples: {100 * accuracy:.2f}%')

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    j = 0
    for image in images_to_plot:
        if j < 9:
            ax = axes[j // 3, j % 3]
            ax.imshow(image.squeeze(), cmap='gray')
            ax.set_title('Prediction: ' + str(predictions[j].item()))
            ax.axis('off')
            j += 1
    plt.tight_layout()
    plt.show(block = False)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        lambda x: 1 - x  # Invert colors
    ])

    predictions = []
    images_to_plot = []

    # Iterate over handwritten digit images
    for i in range(10):
        img_path = f'data/handwritten_images/{i}.jpg'
        if os.path.exists(img_path):
            # Read image
            img = cv2.imread(img_path)
            # Convert image to tensor and apply transformations
            img_tensor = transform(img)
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            # Forward pass
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.item())
                images_to_plot.append(img_tensor.squeeze().numpy())
                print('Correct label:', i)
                print('Output values:', ['%.2f' % val for val in output.squeeze().tolist()])
                print('Index of max output value:', predicted.item())
                print()
        else:
            print(f"Image {i}.jpg not found.")

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    j = 0
    for image in images_to_plot:
        if j < 9:
            ax = axes[j // 3, j % 3]
            ax.imshow(image.squeeze(), cmap='gray')
            ax.set_title('Prediction: ' + str(predictions[j]))
            ax.axis('off')
            j += 1
    plt.tight_layout()
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)