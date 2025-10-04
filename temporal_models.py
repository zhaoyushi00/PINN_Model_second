import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """裁掉卷积产生的多余时间步，保持输入长度。"""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None
        self.final_relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: Iterable[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_channels = list(num_channels)
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation_size
            block = TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=padding,
                dropout=dropout,
            )
            layers.append(block)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNRegressor(nn.Module):
    """
    将输入序列映射到未来一个或多个时间步的 POD 系数预测。

    输入形状： [batch, seq_len, input_dim]
    输出形状： [batch, horizon, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int = 1,
        channels: Optional[Iterable[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = list(channels or [128, 128, 128])
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        last_dim = channels[-1]
        self.readout = nn.Linear(last_dim, horizon * output_dim)
        self.horizon = horizon
        self.output_dim = output_dim

        nn.init.xavier_uniform_(self.readout.weight)
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, D, L]
        x = seq.transpose(1, 2)
        features = self.tcn(x)  # [B, C, L]
        last = features[:, :, -1]
        out = self.readout(last)
        out = out.view(seq.size(0), self.horizon, self.output_dim)
        return out


def append_condition(seq: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """将条件向量扩展到时间维度并拼接到输入序列上。"""
    if cond.dim() == 2:
        cond = cond.unsqueeze(1)
    cond = cond.expand(-1, seq.size(1), -1)
    return torch.cat([seq, cond], dim=-1)

