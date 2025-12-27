#!/usr/bin/env python3
"""Generate signature figures for the paper.

Produces:
- FAR/h vs TTD curves (dynamic-only vs UGF, ID vs OOD)
- Gate vs uncertainty scatter plots
- Timeline plots with intervals, predictions, uncertainty, gate
- Reliability diagrams for calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style settings for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def plot_far_vs_ttd(
    results: pd.DataFrame,
    out_path: Path,
    title: str = "FAR/h vs Time-to-Detect",
) -> None:
    """Plot FAR/h vs TTD for different methods."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = {
        "static_only": "#999999",
        "dynamic_only": "#1f77b4",
        "UGF": "#d62728",
        "late_fusion_avg": "#2ca02c",
        "late_fusion_learned": "#17becf",
        "heuristic_gate": "#ff7f0e",
    }
    
    markers = {
        "static_only": "s",
        "dynamic_only": "^",
        "UGF": "o",
        "late_fusion_avg": "D",
        "late_fusion_learned": "v",
        "heuristic_gate": "P",
    }
    
    for _, row in results.iterrows():
        method = row["method"]
        if method not in colors:
            continue
        ax.scatter(
            row["ttd_median_s"],
            row["far_per_hour"],
            c=colors[method],
            marker=markers[method],
            s=100,
            label=method,
            edgecolors="black",
            linewidths=0.5,
        )
    
    ax.set_xlabel("Time-to-Detect (median, seconds)")
    ax.set_ylabel("False Alarms per Hour")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_gate_vs_uncertainty(
    predictions: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "Gate Value vs Dynamic Uncertainty",
) -> None:
    """Scatter plot of gate values vs dynamic uncertainty."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    g = predictions["g"]
    u_d = predictions["u_d"]
    y = predictions["y"]
    
    # Plot positive and negative separately
    pos_mask = y == 1
    neg_mask = y == 0
    
    ax.scatter(
        u_d[neg_mask], g[neg_mask],
        c="#2ca02c", alpha=0.4, s=20, label="Benign", edgecolors="none"
    )
    ax.scatter(
        u_d[pos_mask], g[pos_mask],
        c="#d62728", alpha=0.6, s=20, label="Trojan", edgecolors="none"
    )
    
    ax.set_xlabel("Dynamic Uncertainty ($u_d$)")
    ax.set_ylabel("Gate Value ($g$)")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, max(0.5, u_d.max() * 1.1))
    ax.set_ylim(0, 1)
    
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_timeline(
    predictions: Dict[str, np.ndarray],
    t_centers: np.ndarray,
    intervals: List[Tuple[float, float]],
    out_path: Path,
    run_id: str = "",
    max_samples: int = 500,
) -> None:
    """Timeline plot showing predictions, uncertainty, and gate over time."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    # Subsample if too many points
    if len(t_centers) > max_samples:
        idx = np.linspace(0, len(t_centers) - 1, max_samples).astype(int)
        t = t_centers[idx]
        p_s = predictions["p_s"][idx]
        p_d = predictions["p_d"][idx]
        p_f = predictions["p_f"][idx]
        u_d = predictions["u_d"][idx]
        g = predictions["g"][idx]
    else:
        t = t_centers
        p_s = predictions["p_s"]
        p_d = predictions["p_d"]
        p_f = predictions["p_f"]
        u_d = predictions["u_d"]
        g = predictions["g"]
    
    # Plot 1: Predictions
    axes[0].plot(t, p_s, label="$p_s$ (static)", color="#999999", linewidth=1)
    axes[0].plot(t, p_d, label="$p_d$ (dynamic)", color="#1f77b4", linewidth=1)
    axes[0].plot(t, p_f, label="$p_f$ (UGF)", color="#d62728", linewidth=1.5)
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper right", ncol=3)
    
    # Shade intervals
    for start, end in intervals:
        for ax in axes:
            ax.axvspan(start, end, alpha=0.2, color="red")
    
    # Plot 2: Uncertainty
    axes[1].plot(t, u_d, label="$u_d$ (dynamic)", color="#ff7f0e", linewidth=1)
    axes[1].set_ylabel("Uncertainty")
    axes[1].legend(loc="upper right")
    
    # Plot 3: Gate
    axes[2].plot(t, g, label="$g$ (gate)", color="#9467bd", linewidth=1)
    axes[2].set_ylabel("Gate Value")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right")
    
    # Plot 4: Binary predictions
    threshold = 0.5
    axes[3].fill_between(t, 0, (p_f > threshold).astype(float), 
                         alpha=0.5, color="#d62728", label="UGF Alert")
    axes[3].set_ylabel("Alert")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylim(0, 1.2)
    axes[3].legend(loc="upper right")
    
    fig.suptitle(f"Timeline: {run_id}" if run_id else "Timeline")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_reliability_diagram(
    y_true: np.ndarray,
    p: np.ndarray,
    out_path: Path,
    title: str = "Reliability Diagram",
    n_bins: int = 10,
) -> None:
    """Reliability diagram showing calibration."""
    fig, ax = plt.subplots(figsize=(5, 5))
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    confidences = []
    counts = []
    
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if np.any(mask):
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(p[mask]))
            count = int(np.sum(mask))
        else:
            acc = np.nan
            conf = np.nan
            count = 0
        accuracies.append(acc)
        confidences.append(conf)
        counts.append(count)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    
    # Plot actual calibration
    valid = ~np.isnan(accuracies)
    ax.bar(
        np.array(bin_centers)[valid],
        np.array(accuracies)[valid],
        width=0.08,
        alpha=0.7,
        color="#1f77b4",
        edgecolor="black",
        label="Model",
    )
    
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_uncertainty_distribution(
    u_id: np.ndarray,
    u_ood: np.ndarray,
    out_path: Path,
    title: str = "Uncertainty Distribution: ID vs OOD",
) -> None:
    """Compare uncertainty distributions for ID vs OOD data."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    bins = np.linspace(0, max(u_id.max(), u_ood.max()) * 1.1, 30)
    
    ax.hist(u_id, bins=bins, alpha=0.6, color="#2ca02c", label="In-Distribution", density=True)
    ax.hist(u_ood, bins=bins, alpha=0.6, color="#d62728", label="Out-of-Distribution", density=True)
    
    ax.axvline(u_id.mean(), color="#2ca02c", linestyle="--", linewidth=2)
    ax.axvline(u_ood.mean(), color="#d62728", linestyle="--", linewidth=2)
    
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right")
    
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate paper figures")
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory containing evaluation results")
    ap.add_argument("--fusion-dir", type=Path, required=True,
                    help="Directory containing fusion predictions")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for figures")
    ap.add_argument("--split-name", type=str, default="unseen_workload")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = args.results_dir / "results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        
        # FAR vs TTD plot
        plot_far_vs_ttd(
            results,
            args.out_dir / f"far_vs_ttd_{args.split_name}.png",
            title=f"FAR/h vs TTD ({args.split_name})",
        )
    
    # Load predictions for detailed plots
    pred_path = args.fusion_dir / "val_predictions.npz"
    if pred_path.exists():
        predictions = _load_npz(pred_path)
        
        # Gate vs uncertainty
        plot_gate_vs_uncertainty(
            predictions,
            args.out_dir / f"gate_vs_uncertainty_{args.split_name}.png",
            title=f"Gate vs Uncertainty ({args.split_name})",
        )
        
        # Reliability diagram
        plot_reliability_diagram(
            predictions["y"],
            predictions["p_f"],
            args.out_dir / f"reliability_{args.split_name}.png",
            title=f"Reliability Diagram ({args.split_name})",
        )

    print(f"\nFigures saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
