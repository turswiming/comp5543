import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def load_mnist_data(batch_size):
    # load train data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True)

    # load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def visualize_mnist_data(train_loader):
    import matplotlib.pyplot as plt
    import numpy as np

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    #show 10 images
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(str(labels[i].item()))
        plt.axis('off')
    
    plt.savefig('mnist_samples.png')