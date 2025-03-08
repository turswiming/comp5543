import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def main():
    transformer = Transformer(input_dim=1, hidden_dim=128, output_dim=10)
    dummy_input = torch.randn(32, 784, 1)
    output = transformer(dummy_input)
    print("output shape:", output.shape)

if __name__ == '__main__':
    main()