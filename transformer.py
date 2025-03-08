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
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            for x, y in zip(images, labels):
                x = x.to(device)
                y = y.to(device)
                x = x.reshape(1,28,28)
                y_pred = model(x)
                predicted = torch.argmax(y_pred)
                total += 1
                correct += (predicted == y).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    model.train()
def main():
    train_loader, test_loader = load_mnist_data(64)
    visualize_mnist_data(train_loader)
    loss_fn = F.cross_entropy
    model = Transformer(28, 128, 10, 8, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()
    
    for epoch in tqdm(range(max_epoch)):
        loss_sum = []
        for x, y in train_loader:
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
        
        # Evaluate every 500 epochs
        if (epoch + 1) % 5 == 0:
            evaluate_model(model, test_loader)
            pass
    pass

if __name__ == '__main__':
    main()