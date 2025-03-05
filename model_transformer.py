import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Positional_Encoding(nn.Module):
    def __init__(self, d_model,max_len=5000):
        super(Positional_Encoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        mesh = torch.arange(0, d_model).float()
        div_term  = 1/(10000**(2*mesh/d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])
        print("pe.shape",pe.shape)
        self.register_buffer('pe', pe)
        pass

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Multi_Head_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(Multi_Head_Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads ==0, "hidden_dim must can be divided by num_heads"
        self.head_dim = hidden_dim // num_heads
        self.Wq = nn.Linear(input_dim, hidden_dim)
        self.Wk = nn.Linear(input_dim, hidden_dim)
        self.Wv = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        seq_len,_ = x.size()

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q = q.view(seq_len,self.num_heads,self.head_dim).transpose(0,1)
        k = k.view(seq_len,self.num_heads,self.head_dim).transpose(0,1)
        v = v.view(seq_len,self.num_heads,self.head_dim).transpose(0,1)

        
        QK = q @ k.transpose(-2,-1)
        softmaxed = torch.softmax(QK/self.input_dim**0.5,dim=-1)
        print("softmaxed shape",softmaxed.shape)
        attention = softmaxed @ v
        attention = attention.transpose(0,1).contiguous().view(seq_len,self.hidden_dim)

        print("attention shape",attention.shape)
        return self.out(attention)

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.positional_encoding = Positional_Encoding(input_dim)
        pass

    def forward(self, x):
        pass

def main():
    print("test PositionalEncoding----------------")
    input = torch.zeros([10,5])
    print("input shape",input.shape)
    pe = Positional_Encoding(5)
    output = pe(input)
    print(output)
    print("output shape",output.shape)
    print("test single head attention-------------")
    att = Multi_Head_Attention(5,70,13,7)
    output = att(output)
    
    print("output shape",output.shape)


if __name__ == '__main__':
    main()