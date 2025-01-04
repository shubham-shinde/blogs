# Implementing Conformer Model as given in Conformer Paper
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, ff_expansion_factor=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * ff_expansion_factor)
        self.fc2 = nn.Linear(d_model * ff_expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = Swish()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.dropout(out)

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out, _ = self.mha(x, x, x)
        out = self.dropout(out)
        return self.layer_norm(residual + out)

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, d_model)
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # Shape: (batch_size, d_model, seq_len)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # Shape: (batch_size, seq_len, d_model)
        return residual + x

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_expansion_factor=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffm1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.mha = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffm2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.mha(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffm2(x)
        return self.layer_norm(x)

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, d_model, num_heads, ff_expansion_factor=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_expansion_factor, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)

# Example usage
if __name__ == "__main__":
    batch_size, seq_len, input_dim = 8, 128, 80
    num_layers, d_model, num_heads = 6, 256, 4

    model = ConformerEncoder(input_dim, num_layers, d_model, num_heads)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    print(out.shape)  # Expected shape: (batch_size, seq_len, d_model)
