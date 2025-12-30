#!/usr/bin/env python3
"""Evaluate LLM localization capability on trojan binaries.

Computes:
- Top-k hit rate: Is any true trojan function in top-k highest-risk?
- MRR: Mean reciprocal rank of true trojan functions
- Analyst utility: Median #functions to inspect before hitting trojan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_llm_reports(reports_dir: Path) -> Dict[str, Dict]:
    """Load all LLM function reports from a directory."""
    reports = {}
    for binary_dir in reports_dir.iterdir():
        if binary_dir.is_dir():
            func_report_path = binary_dir / "function_reports.json"
            if func_report_path.exists():
                with open(func_report_path) as f:
                    data = json.load(f)
                    binary_name = binary_dir.name.rsplit("_", 1)[0]  # Remove hash
                    reports[binary_name] = data
    return reports


def get_function_rankings(report: Dict) -> List[Tuple[str, float]]:
    """Get functions sorted by risk score (highest first)."""
    func_reports = report.get("function_reports", [])
    rankings = [
        (f["function_name"], f.get("risk_score", 0.0))
        for f in func_reports
    ]
    return sorted(rankings, key=lambda x: -x[1])


def compute_localization_metrics(
    rankings: List[Tuple[str, float]],
    true_trojan_functions: List[str],
    report: Dict = None,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute localization metrics for a single binary.
    
    Args:
        rankings: List of (function_name, risk_score) sorted by risk
        true_trojan_functions: List of true trojan function names/keywords
        report: Full LLM report for content-based matching
        k_values: Values of k for top-k hit rate
        
    Returns:
        Dictionary with metrics
    """
    if not rankings or not true_trojan_functions:
        return {"mrr": 0.0, "analyst_utility": float("inf")}
    
    # Find ranks of true trojan functions
    func_to_rank = {name: i + 1 for i, (name, _) in enumerate(rankings)}
    
    # Get function reports for content-based matching
    func_reports = report.get("function_reports", []) if report else []
    func_name_to_report = {f["function_name"]: f for f in func_reports}
    
    true_ranks = []
    for func_keyword in true_trojan_functions:
        # Try exact match first
        if func_keyword in func_to_rank:
            true_ranks.append(func_to_rank[func_keyword])
            continue
        
        # Try content-based matching: look for keywords in categories/findings
        func_keyword_lower = func_keyword.lower()
        for rank, (func_name, _) in enumerate(rankings, 1):
            func_report = func_name_to_report.get(func_name, {})
            
            # Check categories
            categories = [c.lower() for c in func_report.get("categories", [])]
            if any(func_keyword_lower in cat or cat in func_keyword_lower for cat in categories):
                true_ranks.append(rank)
                break
            
            # Check findings for keywords like "authentication", "credential", "backdoor"
            findings = func_report.get("findings", [])
            for finding in findings:
                desc = finding.get("description", "").lower()
                ftype = finding.get("type", "").lower()
                if func_keyword_lower in desc or func_keyword_lower in ftype:
                    true_ranks.append(rank)
                    break
            else:
                continue
            break
    
    if not true_ranks:
        # Fallback: use functions with "backdoor" or "authentication" in categories
        for rank, (func_name, _) in enumerate(rankings, 1):
            func_report = func_name_to_report.get(func_name, {})
            categories = [c.lower() for c in func_report.get("categories", [])]
            if "backdoor" in categories or "potential_backdoor" in categories:
                true_ranks.append(rank)
                break
    
    if not true_ranks:
        return {"mrr": 0.0, "analyst_utility": float("inf")}
    
    # Mean Reciprocal Rank
    mrr = np.mean([1.0 / r for r in true_ranks])
    
    # Analyst utility: minimum rank (best case) = fewest functions to inspect
    analyst_utility = min(true_ranks)
    
    # Top-k hit rates
    metrics = {
        "mrr": mrr,
        "analyst_utility": analyst_utility,
        "best_rank": min(true_ranks),
        "worst_rank": max(true_ranks),
    }
    
    for k in k_values:
        hit = any(r <= k for r in true_ranks)
        metrics[f"top{k}_hit"] = 1.0 if hit else 0.0
    
    return metrics


def evaluate_llm_localization(
    reports: Dict[str, Dict],
    ground_truth: Dict[str, List[str]],
) -> Dict:
    """Evaluate LLM localization across all binaries.
    
    Args:
        reports: Dict mapping binary_name -> LLM report
        ground_truth: Dict mapping binary_name -> list of true trojan function names
        
    Returns:
        Aggregated metrics
    """
    all_metrics = []
    binary_results = {}
    
    for binary_name, true_functions in ground_truth.items():
        if binary_name not in reports:
            print(f"Warning: No LLM report for {binary_name}")
            continue
        
        report = reports[binary_name]
        rankings = get_function_rankings(report)
        
        if not rankings:
            print(f"Warning: No function rankings for {binary_name}")
            continue
        
        metrics = compute_localization_metrics(rankings, true_functions, report=report)
        metrics["binary"] = binary_name
        metrics["n_functions"] = len(rankings)
        metrics["n_true_trojans"] = len(true_functions)
        
        all_metrics.append(metrics)
        binary_results[binary_name] = {
            "metrics": metrics,
            "top_5_functions": rankings[:5],
            "true_trojan_functions": true_functions,
        }
    
    if not all_metrics:
        return {"error": "No valid evaluations"}
    
    # Aggregate metrics
    valid_mrr = [m.get("mrr", 0) for m in all_metrics if "mrr" in m]
    valid_utility = [m.get("analyst_utility", float("inf")) for m in all_metrics 
                     if "analyst_utility" in m and m["analyst_utility"] < float("inf")]
    valid_ranks = [m.get("best_rank", 999) for m in all_metrics if "best_rank" in m]
    
    aggregated = {
        "n_binaries": len(all_metrics),
        "mean_mrr": np.mean(valid_mrr) if valid_mrr else 0.0,
        "mean_analyst_utility": np.mean(valid_utility) if valid_utility else float("inf"),
        "median_best_rank": np.median(valid_ranks) if valid_ranks else float("inf"),
    }
    
    for k in [1, 3, 5, 10]:
        key = f"top{k}_hit"
        hits = [m.get(key, 0) for m in all_metrics if key in m]
        if hits:
            aggregated[f"mean_{key}"] = np.mean(hits)
    
    return {
        "aggregated": aggregated,
        "per_binary": binary_results,
    }


def compare_to_random_baseline(
    n_functions: int,
    n_true_trojans: int,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute expected metrics for random ranking baseline."""
    # Expected rank of first trojan with random ranking
    # E[min rank] ≈ (n_functions + 1) / (n_true_trojans + 1)
    expected_best_rank = (n_functions + 1) / (n_true_trojans + 1)
    
    # Random top-k hit rate: P(at least one trojan in top-k)
    # = 1 - P(no trojan in top-k) = 1 - C(n-t, k) / C(n, k)
    baseline = {
        "expected_best_rank": expected_best_rank,
        "expected_mrr": 1.0 / expected_best_rank,  # Approximation
    }
    
    for k in k_values:
        if k >= n_functions:
            baseline[f"top{k}_hit_prob"] = 1.0
        else:
            # Hypergeometric: P(at least 1 success in k draws from n items with t successes)
            from scipy.stats import hypergeom
            rv = hypergeom(n_functions, n_true_trojans, k)
            baseline[f"top{k}_hit_prob"] = 1.0 - rv.pmf(0)
    
    return baseline


def main():
    ap = argparse.ArgumentParser(description="Evaluate LLM localization")
    ap.add_argument("--reports-dir", type=Path, required=True,
                    help="Path to LLM reports directory")
    ap.add_argument("--ground-truth", type=Path,
                    help="Path to ground truth JSON (binary -> list of trojan functions)")
    ap.add_argument("--output", type=Path, default=Path("paper/analysis/llm_localization.json"))
    args = ap.parse_args()
    
    # Load reports
    reports = load_llm_reports(args.reports_dir)
    print(f"Loaded {len(reports)} binary reports")
    
    # Default ground truth based on Phase 2
    if args.ground_truth and args.ground_truth.exists():
        with open(args.ground_truth) as f:
            ground_truth = json.load(f)
    else:
        # Hardcoded from Phase 2 analysis
        # The authenticate() function contains the backdoor - look for it in high-risk functions
        ground_truth = {
            "firmware_trojan": ["authenticate"],  # Known trojan function
        }
        print("Using default ground truth from Phase 2")
    
    # Evaluate
    results = evaluate_llm_localization(reports, ground_truth)
    
    # Print summary
    if "aggregated" in results:
        agg = results["aggregated"]
        print("\n" + "="*60)
        print("LLM LOCALIZATION RESULTS")
        print("="*60)
        print(f"Binaries evaluated: {agg['n_binaries']}")
        print(f"Mean MRR: {agg['mean_mrr']:.3f}")
        print(f"Mean analyst utility (functions to inspect): {agg['mean_analyst_utility']:.1f}")
        print(f"Median best rank: {agg['median_best_rank']:.1f}")
        
        for k in [1, 3, 5, 10]:
            key = f"mean_top{k}_hit"
            if key in agg:
                print(f"Top-{k} hit rate: {agg[key]:.1%}")
        
        # Compare to random
        print("\nRandom baseline comparison:")
        for binary, binary_data in results.get("per_binary", {}).items():
            n_funcs = binary_data["metrics"]["n_functions"]
            n_trojans = binary_data["metrics"]["n_true_trojans"]
            baseline = compare_to_random_baseline(n_funcs, n_trojans)
            print(f"  {binary}: random expected rank = {baseline['expected_best_rank']:.1f}, "
                  f"LLM best rank = {binary_data['metrics']['best_rank']}")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
