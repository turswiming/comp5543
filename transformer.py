# this is a self implementation of transformer model
# author: Ziqi Li
# for COMP5543
# date: 2025-03-06

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mnist_dataset import load_mnist_data, visualize_mnist_data
from model_transformer import Transformer
from tqdm import tqdm
max_epoch = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate_model(model, test_loader,criterion):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(-1,28,28)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            loss = criterion(y_pred, y)
            losses.append(loss.item())
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    model.train()
    return accuracy, np.mean(losses)
def main():
    train_loader, test_loader = load_mnist_data(64)
    visualize_mnist_data(train_loader)
    loss_fn = F.cross_entropy
    model = Transformer(28, 64, 10, 2, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()
    loss_epoch_train = []
    loss_epoch_test = []
    acc_epoch_train = []
    for epoch in tqdm(range(max_epoch)):
        loss_sum = []
        for x, y in tqdm(train_loader):
            x = x.reshape(-1, 28, 28)
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss_sum.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch + 1}/{max_epoch}, Loss: {np.mean(loss_sum)}')
        loss_epoch_train.append(np.mean(loss_sum))
        # Evaluate every 500 epochs
        if (epoch + 1) % 1 == 0:
            acc ,loss_eval= evaluate_model(model, test_loader,loss_fn)
            loss_epoch_test.append(loss_eval)
            acc_epoch_train.append(acc)
            pass

    #visualize the loss and accuracy
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].plot(loss_epoch_train, label='train')
    axs[0].plot(loss_epoch_test, label='test')
    axs[0].set_title('Loss')
    axs[0].legend()
    
    axs[1].plot(acc_epoch_train, label='train')
    axs[1].set_title('Accuracy')
    axs[1].legend()
    plt.savefig('loss_acc.png')
    pass

if __name__ == '__main__':
    main()