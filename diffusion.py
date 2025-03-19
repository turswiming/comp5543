
# this is a self implementation of diffusion model
# author: Ziqi Li
# for COMP5543
# date: 2025-03-09

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

max_epoch = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_cifar_data(batch_size):
    # load train data
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Resize(32),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            transforms.RandomHorizontalFlip(),
                       ])),
        batch_size=batch_size, shuffle=True)

    # load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
        ])),
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def visualize_cifar_data(train_loader):
    import matplotlib.pyplot as plt
    import numpy as np

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print("image shape",images.shape)
    print("image range",images.min(), images.max())
    #show 10 images
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().squeeze().transpose(1,2,0))
        plt.title(str(labels[i].item()))
        plt.axis('off')
    
    plt.savefig('cifar_samples.png')

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.freq = 10 ** (torch.arange(0, dim, 2).float() / dim)

    def forward(self, t):
        # t: [batch_size]
        t = t.view(-1, 1) * self.freq.to(t.device)
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)

class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    def forward(self, x, t):
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        return h

class upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    def forward(self, x, t):
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        return h

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super(Unet, self).__init__()
        self.time_embedding = TimeEmbedding(in_channels)
        self.encoder = nn.ModuleList([
            downsample(in_channels, hidden_channels),
            downsample(hidden_channels, hidden_channels*2),
            downsample(hidden_channels*2, hidden_channels*4),
            downsample(hidden_channels*4, hidden_channels*8),
            downsample(hidden_channels*8, hidden_channels*16),
            downsample(hidden_channels*16, hidden_channels*32),
        ])
        self.middle = nn.Sequential(
            nn.Conv2d(hidden_channels*32, hidden_channels*32, 3, padding=1),
            nn.BatchNorm2d(hidden_channels*32),
            nn.SiLU(),
        )
        self.decoder = nn.ModuleList([
            upsample(hidden_channels*32, hidden_channels*16),
            upsample(hidden_channels*16, hidden_channels*8),
            upsample(hidden_channels*8, hidden_channels*4),
            upsample(hidden_channels*4, hidden_channels*2),
            upsample(hidden_channels*2, hidden_channels),
            upsample(hidden_channels, hidden_channels),
        ])
        self.output = nn.Conv2d(hidden_channels, out_channels, 1)
    def forward(self, x, t):
        t = self.time_embedding(t)
        h = x
        hs = []
        for layer in self.encoder:
            h = layer(h, t)
            hs.append(h)
        h = self.middle(h)
        for layer in self.decoder:
            h = layer(h, t)
            h = h + hs.pop()
        
        h = self.output(h)
        return h

class DiffusionModel:
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=64, gen_steps = 10):
        self.model = Unet(in_channels, out_channels, hidden_channels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()# this is L2
        self.gen_steps = gen_steps
        import datetime
        import os
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.timestamp = timestamp
        self.log_dir = os.path.join('logs', 'diffusion', timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0

    def forward(self, x, t):
        return self.model(x, t)
    def linear_beta_schedule(self,num_steps=1000, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, num_steps)
    def forward_diffusion(self, x0, t):
        alpha_t = t/self.gen_steps
        
        #boardcast alpha_t to x0
        alpha_t = alpha_t.view(-1,1,1,1) 
        self.writer.add_histogram('alpha_t', alpha_t)
        noise = torch.randn_like(x0)*0.3
        #clamp noise to [-1,1]
        noise = torch.clamp(noise,-1,1)
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1-alpha_t) * noise
        return xt

    def train(self, train_loader, epochs=10):
        self.model.to(device)
        self.model.train()
        for epoch in tqdm(range(epochs)):
            batch_idx = 0
            for data, target in tqdm(train_loader):
                batch_idx += 1
                self.global_step += 1
                x = data.to(device)
                t = torch.randint(1, self.gen_steps, (x.size(0),)).to(device)  # 随机采样时间步
                t = t.float().to(device)
                #convert [1,256] to [256,1]
                t = t.unsqueeze(1)

                xt = self.forward_diffusion(x, t)
                xtp1 = self.forward_diffusion(x, t+1)
                pred_noise = self.model(xt, t)
                self.writer.add_histogram('pred_noise', pred_noise, self.global_step)
                loss = self.criterion(pred_noise, xtp1)
                self.writer.add_histogram('xt', xt, self.global_step)
                self.writer.add_histogram('x', x, self.global_step)
                self.writer.add_scalar('loss', loss.item(), self.global_step)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss', loss.item(), self.global_step)
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss}')
                    losses = []
                if batch_idx % 500 == 0:
                    generated_images = self.generate(10)

                    visualize_generated_data(generated_images,f'Epoch{epoch}Batch{batch_idx}.png')

    def generate(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            noise = torch.normal(0, 0.5, size=(num_samples, 3, 32, 32))
            #clamp noise to [-1,1]
            noise = torch.clamp(noise,-1,1)
            noise = noise.to(device)
            for step in range(self.gen_steps):
                t = torch.full((num_samples, 1), step,dtype=torch.float32).to(device)
                t/=float(self.gen_steps)
                noise = self.forward(noise, t)
            return noise



def visualize_generated_data(generated_images,filepath):
    import matplotlib.pyplot as plt
    import numpy as np

    #show 10 images
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(generated_images[i].cpu().detach().numpy().squeeze().transpose(1,2,0)*0.5+0.5)
        plt.axis('off')
    
    plt.savefig(filepath)

def main():
    train_loader, test_loader = load_cifar_data(16)
    visualize_cifar_data(train_loader)
    model = DiffusionModel(gen_steps=10)
    model.train(train_loader, epochs=max_epoch, )
    generated_images = model.generate(10)
    print(generated_images.shape)
    # visualize_generated_data(generated_images)

if __name__ == '__main__':
    main()
