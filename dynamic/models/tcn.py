#!/usr/bin/env python3
"""Temporal Convolutional Network (TCN) for dynamic telemetry classification.

A causal 1D CNN with dilated convolutions and residual connections.
Designed for window-level classification of telemetry sequences.

References:
- Bai, Kolter, Koltun. "An Empirical Evaluation of Generic Convolutional 
  and Recurrent Networks for Sequence Modeling", 2018.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-side padding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual block with two causal convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (with 1x1 conv if channels change)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + residual)
        return out


class TCN(nn.Module):
    """Temporal Convolutional Network for sequence classification.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of channels in TCN layers
        num_layers: Number of temporal blocks
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Build TCN layers with exponential dilation
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            layers.append(
                TemporalBlock(in_ch, hidden_size, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*layers)
        
        # Classification head (global average pooling + linear)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F] tensor where T=timesteps, F=features
            
        Returns:
            logits: [B] tensor of logits for binary classification
        """
        # Transpose to [B, F, T] for 1D conv
        x = x.transpose(1, 2)
        
        # TCN
        out = self.tcn(x)  # [B, H, T]
        
        # Global average pooling over time
        out = out.mean(dim=2)  # [B, H]
        
        # Classification
        logits = self.classifier(out).squeeze(-1)  # [B]
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = x.transpose(1, 2)
        out = self.tcn(x)
        return out.mean(dim=2)


class SimpleCNN(nn.Module):
    """Simple 1D CNN baseline for comparison."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 5, padding=2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 5, padding=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.mean(dim=2)  # Global average pooling
        return self.classifier(x).squeeze(-1)
