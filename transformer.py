# this is a self implementation of transformer model
# author: Ziqi Li
# for COMP5543
# date: 2025-03-06

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms


max_epoch = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(Positional_Encoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class Multi_Head_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(Multi_Head_Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        
        self.Wq = nn.Linear(input_dim, hidden_dim)
        self.Wk = nn.Linear(input_dim, hidden_dim)
        self.Wv = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        
        for layer in [self.Wq, self.Wk, self.Wv]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        B, seq_len, _ = x.size()
        
        
        q = self.Wq(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scaling_factor = self.head_dim ** 0.5
        attn_scores = (q @ k.transpose(-2, -1)) / scaling_factor
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = attn_weights @ v
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_dim)
        return self.out(attn_output)

class Feed_Forward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Feed_Forward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class Encoder_Transformer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, dropout=0.1):
        super(Encoder_Transformer, self).__init__()
        self.attention = Multi_Head_Attention(d_model, hidden_dim, d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = Feed_Forward(d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=10, 
                 num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.up_layer = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = Positional_Encoding(hidden_dim)
        self.encoders = nn.ModuleList([
            Encoder_Transformer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        nn.init.xavier_uniform_(self.up_layer.weight)
        nn.init.zeros_(self.up_layer.bias)
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, x):
        x = self.up_layer(x)
        x = self.pos_encoder(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


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