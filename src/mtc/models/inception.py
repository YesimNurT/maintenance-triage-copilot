"""InceptionTime for multivariate time series (Fawaz et al., 2020), PyTorch.

Each block runs three convolutions with different kernel lengths plus a max-pool branch
in parallel, so the network sees short and long patterns at once. Residual shortcuts
every three blocks; global average pooling makes it independent of sequence length.
"""

import torch
from torch import nn

KERNEL_SIZES = (39, 19, 9)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int, bottleneck: int = 32):
        super().__init__()
        use_bottleneck = in_channels > 1
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck, 1, bias=False) if use_bottleneck else nn.Identity()
        )
        conv_in = bottleneck if use_bottleneck else in_channels
        self.convs = nn.ModuleList(
            nn.Conv1d(conv_in, filters, k, padding=k // 2, bias=False) for k in KERNEL_SIZES
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, filters, 1, bias=False)
        )
        self.norm = nn.BatchNorm1d(filters * (len(KERNEL_SIZES) + 1))
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        branches = [conv(z) for conv in self.convs] + [self.pool(x)]
        return self.act(self.norm(torch.cat(branches, dim=1)))


class InceptionTime(nn.Module):
    """Input ``[batch, channels, time]``, output one logit per sequence."""

    def __init__(self, in_channels: int, filters: int = 32, depth: int = 6):
        super().__init__()
        width = filters * (len(KERNEL_SIZES) + 1)
        self.blocks = nn.ModuleList(
            InceptionBlock(in_channels if i == 0 else width, filters) for i in range(depth)
        )
        self.shortcuts = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(in_channels if i == 0 else width, width, 1, bias=False),
                nn.BatchNorm1d(width),
            )
            for i in range(0, depth, 3)
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 2 or i == len(self.blocks) - 1:
                x = torch.relu(x + self.shortcuts[i // 3](residual))
                residual = x
        return self.head(x.mean(dim=2)).squeeze(1)
