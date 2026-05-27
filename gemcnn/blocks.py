import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    Single 1-D residual block: Conv→BN→ReLU→Conv→BN + skip→ReLU.
    Strided 1×1 conv on skip path when stride>1 or channels change.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 17, stride: int = 2):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, stride=stride,
                               padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size, stride=1,
                               padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        needs_proj = (stride != 1 or in_channels != out_channels)
        self.skip  = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if needs_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class MultiHeadSelfAttention(nn.Module):
    """Wrapped nn.MultiheadAttention with pre-norm residual."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


class FeedForward(nn.Module):
    """Position-wise FFN with residual + LayerNorm."""
    def __init__(self, dim: int, ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SAINTBlock(nn.Module):
    """
    One SAINT block (Both mode):
      intra-sample attention → FFN → inter-sample attention → FFN.
    """
    def __init__(self, dim: int, num_heads: int = 4,
                 ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.intra_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.intra_ff   = FeedForward(dim, ff_dim, dropout)
        self.inter_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.inter_ff   = FeedForward(dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_features, dim)

        # intra-sample: attend across features within each sample
        x = self.intra_ff(self.intra_attn(x))

        # inter-sample: attend across batch for each feature position
        x = x.permute(1, 0, 2)                # (F, B, dim)
        x = self.inter_ff(self.inter_attn(x))
        x = x.permute(1, 0, 2)                # (B, F, dim)
        return x
