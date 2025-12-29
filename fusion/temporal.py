#!/usr/bin/env python3
"""Temporal smoothing for window-level predictions.

Raw window-wise probabilities are noisy. A small amount of sequential logic
can massively improve FAR without hurting TTD:
- Debounce: Require k consecutive windows above threshold
- HMM filter: Bayesian filtering over active/inactive states
- CUSUM: Change detection on log-odds

Reference: Standard practice in online detection systems.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def debounce(
    p_windows: np.ndarray,
    threshold: float,
    k_consecutive: int = 2,
) -> np.ndarray:
    """Require k consecutive windows above threshold to trigger detection.
    
    This reduces false alarms from isolated spurious high predictions.
    
    Args:
        p_windows: Window probabilities [N]
        threshold: Detection threshold
        k_consecutive: Number of consecutive windows required
        
    Returns:
        Binary detection signal [N]
    """
    above = (p_windows >= threshold).astype(np.int32)
    
    if k_consecutive <= 1:
        return above
    
    # Use convolution to count consecutive windows
    kernel = np.ones(k_consecutive, dtype=np.int32)
    
    # convolve with 'full' mode, then align to get causal output
    conv = np.convolve(above, kernel, mode='full')[:len(above)]
    
    # Detection fires when we have k consecutive
    detection = (conv >= k_consecutive).astype(np.int32)
    
    return detection


def debounce_with_events(
    p_windows: np.ndarray,
    t_centers: np.ndarray,
    threshold: float,
    k_consecutive: int = 2,
    window_len_s: float = 5.0,
) -> List[Tuple[float, float]]:
    """Debounce and return event intervals.
    
    Returns:
        List of (start_time, end_time) tuples for detected events
    """
    detection = debounce(p_windows, threshold, k_consecutive)
    
    events = []
    half = window_len_s / 2
    
    in_event = False
    event_start = 0.0
    
    for i, (t, d) in enumerate(zip(t_centers, detection)):
        if d == 1 and not in_event:
            # Start new event
            in_event = True
            event_start = t - half
        elif d == 0 and in_event:
            # End event
            in_event = False
            events.append((event_start, t_centers[i-1] + half))
    
    # Handle event that extends to end
    if in_event:
        events.append((event_start, t_centers[-1] + half))
    
    return events


def exponential_moving_average(
    p_windows: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """Apply exponential moving average smoothing.
    
    Args:
        p_windows: Window probabilities [N]
        alpha: Smoothing factor (higher = less smoothing)
        
    Returns:
        Smoothed probabilities [N]
    """
    smoothed = np.zeros_like(p_windows)
    smoothed[0] = p_windows[0]
    
    for i in range(1, len(p_windows)):
        smoothed[i] = alpha * p_windows[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def hmm_filter(
    p_windows: np.ndarray,
    p_transition: float = 0.02,
    prior_active: float = 0.18,
) -> np.ndarray:
    """Simple HMM-style forward filter over active/inactive states.
    
    Assumes:
    - Two states: inactive (0), active (1)
    - Symmetric transition probability
    - Observation is p_d directly (treated as P(obs|active))
    
    Args:
        p_windows: Window probabilities from model [N]
        p_transition: Probability of state change per window
        prior_active: Prior probability of active state
        
    Returns:
        Filtered state probabilities [N]
    """
    n = len(p_windows)
    filtered = np.zeros(n)
    
    # Transition matrix
    # P(stay) = 1 - p_transition
    # P(switch) = p_transition
    
    # Initial belief
    belief_active = prior_active
    
    for i in range(n):
        # Observation likelihood
        # P(high_p | active) ≈ p_d
        # P(high_p | inactive) ≈ 1 - p_d (inverted)
        p_obs = p_windows[i]
        
        # Update with observation (Bayes rule)
        likelihood_active = p_obs
        likelihood_inactive = 1 - p_obs
        
        posterior_active = likelihood_active * belief_active
        posterior_inactive = likelihood_inactive * (1 - belief_active)
        
        # Normalize
        total = posterior_active + posterior_inactive
        if total > 0:
            belief_active = posterior_active / total
        
        filtered[i] = belief_active
        
        # Predict next step (apply transition)
        belief_active = (
            belief_active * (1 - p_transition) +  # Stay active
            (1 - belief_active) * p_transition    # Switch to active
        )
    
    return filtered


def cusum(
    p_windows: np.ndarray,
    threshold: float = 0.3,
    drift: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """CUSUM change detection on log-odds.
    
    Detects upward shifts in probability that indicate activation.
    
    Args:
        p_windows: Window probabilities [N]
        threshold: Detection threshold for CUSUM statistic
        drift: Expected drift under null hypothesis
        
    Returns:
        cusum_stat: CUSUM statistic [N]
        detection: Binary detection signal [N]
    """
    # Convert to log-odds
    eps = 1e-6
    p_clipped = np.clip(p_windows, eps, 1 - eps)
    log_odds = np.log(p_clipped / (1 - p_clipped))
    
    # CUSUM for detecting upward shift
    cusum_stat = np.zeros_like(log_odds)
    
    for i in range(1, len(log_odds)):
        # Accumulate deviations above drift
        cusum_stat[i] = max(0, cusum_stat[i-1] + log_odds[i] - drift)
    
    detection = (cusum_stat >= threshold).astype(np.int32)
    
    return cusum_stat, detection


def apply_temporal_smoothing(
    p_windows: np.ndarray,
    method: str = "debounce",
    threshold: float = 0.3,
    **kwargs,
) -> np.ndarray:
    """Apply specified temporal smoothing method.
    
    Args:
        p_windows: Window probabilities [N]
        method: One of 'debounce', 'ema', 'hmm', 'cusum'
        threshold: Detection threshold (for methods that need it)
        **kwargs: Method-specific parameters
        
    Returns:
        Smoothed/filtered output [N]
    """
    if method == "debounce":
        k = kwargs.get("k_consecutive", 2)
        return debounce(p_windows, threshold, k).astype(np.float32)
    
    elif method == "ema":
        alpha = kwargs.get("alpha", 0.3)
        return exponential_moving_average(p_windows, alpha)
    
    elif method == "hmm":
        p_trans = kwargs.get("p_transition", 0.02)
        prior = kwargs.get("prior_active", 0.18)
        return hmm_filter(p_windows, p_trans, prior)
    
    elif method == "cusum":
        drift = kwargs.get("drift", 0.1)
        _, detection = cusum(p_windows, threshold, drift)
        return detection.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
