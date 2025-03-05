# this is a self implementation of transformer model
# author: Ziqi Li
# for COMP5543
# date: 2025-03-06

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mnist_dataset import load_mnist_data, visualize_mnist_data
def main():
    print("torch version: ", torch.__version__)
    print("cuda available: ", torch.cuda.is_available())
    print("cuda version: ", torch.version.cuda)
    print("cuda device count: ", torch.cuda.device_count())
    print("cuda device name: ", torch.cuda.get_device_name())
    train_loader, test_loader = load_mnist_data(64)
    visualize_mnist_data(train_loader)
    pass

if __name__ == '__main__':
    main()