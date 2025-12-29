#!/usr/bin/env python3
"""Fusion baseline methods for comparison with UGF.

All methods take static and dynamic predictions/uncertainties
and output a fused prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class FusionResult:
    """Container for fusion outputs."""
    p: np.ndarray  # Fused probabilities [N]
    method: str
    g: Optional[np.ndarray] = None  # Gate values if applicable


def static_only(p_s: np.ndarray, **kwargs) -> FusionResult:
    """Baseline: use only static predictions.
    
    For window-level prediction, p_s is broadcast across windows.
    """
    return FusionResult(p=p_s, method="static_only")


def dynamic_only(p_d: np.ndarray, **kwargs) -> FusionResult:
    """Baseline: use only dynamic predictions."""
    return FusionResult(p=p_d, method="dynamic_only")


def late_fusion_avg(
    p_s: np.ndarray,
    p_d: np.ndarray,
    **kwargs,
) -> FusionResult:
    """Baseline: simple average of predictions."""
    p = 0.5 * p_s + 0.5 * p_d
    return FusionResult(p=p, method="late_fusion_avg")


def late_fusion_learned(
    p_s: np.ndarray,
    p_d: np.ndarray,
    y_train: np.ndarray,
    p_s_train: np.ndarray,
    p_d_train: np.ndarray,
    **kwargs,
) -> FusionResult:
    """Baseline: logistic regression on [p_s, p_d].
    
    Learns optimal weights from training data.
    """
    # Stack features
    X_train = np.column_stack([p_s_train, p_d_train])
    X_test = np.column_stack([p_s, p_d])
    
    # Fit logistic regression
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    
    # Predict
    p = clf.predict_proba(X_test)[:, 1]
    return FusionResult(p=p, method="late_fusion_learned")


def product_of_experts(
    p_s: np.ndarray,
    p_d: np.ndarray,
    **kwargs,
) -> FusionResult:
    """Baseline: product of experts (normalized).
    
    p_f = (p_s * p_d) / Z where Z normalizes probabilities.
    """
    # Clip to avoid zeros
    p_s_c = np.clip(p_s, 1e-7, 1 - 1e-7)
    p_d_c = np.clip(p_d, 1e-7, 1 - 1e-7)
    
    # Product of experts
    p_pos = p_s_c * p_d_c
    p_neg = (1 - p_s_c) * (1 - p_d_c)
    p = p_pos / (p_pos + p_neg)
    
    return FusionResult(p=p, method="product_of_experts")


def logit_add(
    p_s: np.ndarray,
    p_d: np.ndarray,
    **kwargs,
) -> FusionResult:
    """Baseline: add logits then sigmoid.
    
    p_f = σ(logit(p_s) + logit(p_d))
    """
    # Clip and convert to logits
    p_s_c = np.clip(p_s, 1e-7, 1 - 1e-7)
    p_d_c = np.clip(p_d, 1e-7, 1 - 1e-7)
    
    logit_s = np.log(p_s_c / (1 - p_s_c))
    logit_d = np.log(p_d_c / (1 - p_d_c))
    
    # Add logits and apply sigmoid
    p = 1.0 / (1.0 + np.exp(-(logit_s + logit_d)))
    
    return FusionResult(p=p, method="logit_add")


def heuristic_uncertainty_gate(
    p_s: np.ndarray,
    p_d: np.ndarray,
    u_d: np.ndarray,
    alpha: float = 5.0,
    **kwargs,
) -> FusionResult:
    """Baseline: heuristic gate based on dynamic uncertainty.
    
    g = exp(-α * u_d)
    p_f = g * p_d + (1-g) * p_s
    
    When u_d is high (uncertain), gate decreases and falls back to static.
    """
    g = np.exp(-alpha * u_d)
    p = g * p_d + (1 - g) * p_s
    return FusionResult(p=p, method=f"heuristic_gate_alpha{alpha}", g=g)


def heuristic_both_uncertainty_gate(
    p_s: np.ndarray,
    p_d: np.ndarray,
    u_s: np.ndarray,
    u_d: np.ndarray,
    alpha: float = 5.0,
    **kwargs,
) -> FusionResult:
    """Baseline: heuristic gate using both uncertainties.
    
    g = σ(α * (u_s - u_d))
    Trust dynamic when static is more uncertain than dynamic.
    """
    logit = alpha * (u_s - u_d)
    g = 1.0 / (1.0 + np.exp(-logit))
    p = g * p_d + (1 - g) * p_s
    return FusionResult(p=p, method=f"heuristic_both_gate_alpha{alpha}", g=g)


def hierarchical(
    p_s: np.ndarray,
    p_d: np.ndarray,
    u_s: np.ndarray,
    u_d: np.ndarray,
    **kwargs,
) -> FusionResult:
    """Hierarchical fusion: p_active = p_cap × p_act_given_cap
    
    This is the correct probabilistic decomposition:
    - p_s = P(capability) = "Is this binary capable of trojan behavior?"
    - p_d = P(activation | capability) = "Is trojan active NOW?"
    - p_active = p_cap × p_act|cap
    
    The static expert acts as a prior/FAR reducer on benign binaries.
    This is the recommended approach from ML expert review.
    """
    # Multiplicative combination (prior × likelihood)
    p = p_s * p_d
    
    # Uncertainty propagation: std(X*Y) ≈ sqrt((Y*σ_X)² + (X*σ_Y)²)
    u = np.sqrt((p_d * u_s)**2 + (p_s * u_d)**2)
    
    return FusionResult(p=p, method="hierarchical", u=u)


def hierarchical_veto(
    p_s: np.ndarray,
    p_d: np.ndarray,
    u_s: np.ndarray,
    u_d: np.ndarray,
    cap_threshold: float = 0.5,
    **kwargs,
) -> FusionResult:
    """Hierarchical with hard veto: if p_cap < threshold, output 0.
    
    This is a more aggressive version that completely suppresses
    detection on binaries the static expert considers benign.
    """
    # Veto: set p to 0 when capability is low
    veto_mask = (p_s >= cap_threshold).astype(np.float32)
    p = veto_mask * p_d
    
    return FusionResult(p=p, method=f"hierarchical_veto_t{cap_threshold}")


# Registry of all baseline methods
BASELINES = {
    "static_only": static_only,
    "dynamic_only": dynamic_only,
    "late_fusion_avg": late_fusion_avg,
    "late_fusion_learned": late_fusion_learned,
    "product_of_experts": product_of_experts,
    "logit_add": logit_add,
    "heuristic_gate": heuristic_uncertainty_gate,
    "heuristic_both_gate": heuristic_both_uncertainty_gate,
    "hierarchical": hierarchical,
    "hierarchical_veto": hierarchical_veto,
}


def get_baseline(name: str):
    """Get a baseline method by name."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINES.keys())}")
    return BASELINES[name]

