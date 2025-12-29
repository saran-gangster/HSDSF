#!/usr/bin/env python3
"""Hierarchical Fusion: p_active = p_cap × p_act_given_cap

This implements the correct probabilistic decomposition as recommended by
the ML expert. Static and dynamic experts answer different questions:
- Static: P(C=1|x_s) = "Is this binary capable of trojan behavior?"
- Dynamic: P(A_t=1|x_d(t), C=1) = "Is trojan active NOW, given capability?"

Combined: P(active) = P(capability) × P(activation | capability)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class HierarchicalResult:
    """Container for hierarchical fusion outputs."""
    p_active: np.ndarray        # Fused activation probability [N]
    p_cap: np.ndarray           # Capability probability (static) [N]
    p_act_given_cap: np.ndarray # Activation|capability probability (dynamic) [N]
    u_active: Optional[np.ndarray] = None  # Propagated uncertainty [N]


def hierarchical_fusion(
    p_cap: np.ndarray,
    p_act_given_cap: np.ndarray,
    u_cap: Optional[np.ndarray] = None,
    u_act: Optional[np.ndarray] = None,
) -> HierarchicalResult:
    """Hierarchical fusion: p_active = p_cap × p_act_given_cap
    
    This is the correct probabilistic combination when:
    - p_cap estimates capability (binary-level)
    - p_act_given_cap estimates activation given capability (window-level)
    
    The static expert acts as a prior that reduces false alarms on
    benign binaries by pushing p_active toward 0 when p_cap is low.
    
    Args:
        p_cap: Capability probability from static expert [N]
        p_act_given_cap: Activation probability from dynamic expert [N]
        u_cap: Optional uncertainty from static expert [N]
        u_act: Optional uncertainty from dynamic expert [N]
        
    Returns:
        HierarchicalResult with fused predictions and optional uncertainty
    """
    # Multiplicative combination (prior × likelihood)
    p_active = p_cap * p_act_given_cap
    
    # First-order uncertainty propagation if both uncertainties provided
    u_active = None
    if u_cap is not None and u_act is not None:
        # Var(X*Y) ≈ E[Y]²Var(X) + E[X]²Var(Y) for independent X,Y
        # std(X*Y) ≈ sqrt((Y*σ_X)² + (X*σ_Y)²)
        u_active = np.sqrt((p_act_given_cap * u_cap)**2 + (p_cap * u_act)**2)
    
    return HierarchicalResult(
        p_active=p_active,
        p_cap=p_cap,
        p_act_given_cap=p_act_given_cap,
        u_active=u_active,
    )


def hierarchical_fusion_with_threshold_modulation(
    p_cap: np.ndarray,
    p_act_given_cap: np.ndarray,
    base_threshold: float = 0.3,
    cap_weight: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hierarchical fusion with per-binary threshold modulation.
    
    Instead of fixed threshold, adjust threshold based on capability:
    - High capability binaries: lower threshold (more sensitive)
    - Low capability binaries: higher threshold (more conservative)
    
    Args:
        p_cap: Capability probability [N]
        p_act_given_cap: Activation probability [N]
        base_threshold: Base detection threshold
        cap_weight: How much capability affects threshold (0-1)
        
    Returns:
        p_active: Fused probabilities [N]
        thresholds: Per-sample thresholds [N]
    """
    p_active = p_cap * p_act_given_cap
    
    # Modulated threshold: lower for high-cap, higher for low-cap
    # threshold = base * (2 - p_cap) when cap_weight=1
    thresholds = base_threshold * (1 + cap_weight * (1 - p_cap))
    
    return p_active, thresholds


class HierarchicalFusionModule(nn.Module):
    """PyTorch module for hierarchical fusion (for gradient-based analysis)."""
    
    def forward(
        self,
        p_cap: torch.Tensor,
        p_act_given_cap: torch.Tensor,
    ) -> torch.Tensor:
        """Compute p_active = p_cap × p_act_given_cap"""
        return p_cap * p_act_given_cap
    
    def forward_with_uncertainty(
        self,
        p_cap: torch.Tensor,
        p_act: torch.Tensor,
        u_cap: torch.Tensor,
        u_act: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with uncertainty propagation."""
        p_active = p_cap * p_act
        u_active = torch.sqrt((p_act * u_cap)**2 + (p_cap * u_act)**2)
        return p_active, u_active


# Convenience functions for evaluation
def compute_hierarchical_predictions(
    p_s: np.ndarray,
    p_d: np.ndarray,
    u_s: Optional[np.ndarray] = None,
    u_d: Optional[np.ndarray] = None,
) -> HierarchicalResult:
    """Convenience wrapper using existing static/dynamic naming."""
    return hierarchical_fusion(
        p_cap=p_s,
        p_act_given_cap=p_d,
        u_cap=u_s,
        u_act=u_d,
    )
