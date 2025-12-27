#!/usr/bin/env python3
"""Uncertainty-Gated Fusion (UGF) gate model.

The gate learns to route trust between static and dynamic experts
based on their uncertainties and meta-context.

Reference architecture:
  g(t) = σ(f_θ([u_s, u_d(t), m(t)]))
  p_f(t) = g(t)·p_d(t) + (1-g(t))·p_s
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GateInputs:
    """Container for gate input features."""
    u_s: np.ndarray  # Static uncertainty [N]
    u_d: np.ndarray  # Dynamic uncertainty [N]
    mask_summary: Optional[np.ndarray] = None  # Mask features [N, M]
    regime_features: Optional[np.ndarray] = None  # Regime bins [N, R]


class UGFGate(nn.Module):
    """Learned gate for uncertainty-gated fusion.
    
    Inputs:
        - u_s: static expert uncertainty
        - u_d: dynamic expert uncertainty (per window)
        - meta: optional meta-context (mask summaries, regime bins)
    
    Output:
        - g ∈ [0,1]: weight for dynamic expert (1-g for static)
    """
    
    def __init__(
        self,
        n_meta_features: int = 0,
        hidden_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input: [u_s, u_d] + optional meta
        input_size = 2 + n_meta_features
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Initialize gate neutrally (sigmoid(0) = 0.5)
        with torch.no_grad():
            self.net[-1].bias.fill_(0.0)
    
    def forward(
        self,
        u_s: torch.Tensor,
        u_d: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            u_s: [N] static uncertainties
            u_d: [N] dynamic uncertainties
            meta: [N, M] optional meta features
            
        Returns:
            g: [N] gate values in [0,1]
        """
        # Build input
        if u_s.dim() == 1:
            u_s = u_s.unsqueeze(1)
        if u_d.dim() == 1:
            u_d = u_d.unsqueeze(1)
        
        if meta is not None:
            x = torch.cat([u_s, u_d, meta], dim=1)
        else:
            x = torch.cat([u_s, u_d], dim=1)
        
        # Forward through MLP
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)


class UGFFusion(nn.Module):
    """Full UGF fusion model combining gate with expert predictions."""
    
    def __init__(self, gate: UGFGate):
        super().__init__()
        self.gate = gate
    
    def forward(
        self,
        p_s: torch.Tensor,
        p_d: torch.Tensor,
        u_s: torch.Tensor,
        u_d: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            p_s: [N] static predictions
            p_d: [N] dynamic predictions
            u_s: [N] static uncertainties
            u_d: [N] dynamic uncertainties
            meta: [N, M] optional meta features
            
        Returns:
            p_f: [N] fused predictions
        """
        g = self.gate(u_s, u_d, meta)
        p_f = g * p_d + (1 - g) * p_s
        return p_f
    
    def forward_with_gate(
        self,
        p_s: torch.Tensor,
        p_d: torch.Tensor,
        u_s: torch.Tensor,
        u_d: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both prediction and gate value."""
        g = self.gate(u_s, u_d, meta)
        p_f = g * p_d + (1 - g) * p_s
        return p_f, g


def create_gate(n_meta_features: int = 0) -> UGFGate:
    """Factory function for creating a gate model."""
    return UGFGate(n_meta_features=n_meta_features)


def create_fusion_model(n_meta_features: int = 0) -> UGFFusion:
    """Factory function for creating a full fusion model."""
    gate = create_gate(n_meta_features)
    return UGFFusion(gate)
