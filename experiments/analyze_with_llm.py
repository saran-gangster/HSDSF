#!/usr/bin/env python3
"""Analyze synthetic binaries with LLM and compute localization metrics.

This script:
1. Loads synthetic binary disassembly
2. Calls OpenRouter API (MiMo-V2-Flash) to analyze each function
3. Ranks functions by risk_score
4. Computes MRR, Top-k, improvement factor vs ground truth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "xiaomi/mimo-v2-flash:free"


def analyze_function(
    api_key: str,
    model: str,
    function_name: str,
    disassembly: str,
    binary_name: str,
) -> Dict[str, Any]:
    """Analyze a single function with OpenRouter LLM."""
    
    system_prompt = """You are an EXPERT firmware security analyst specialized in reverse engineering ARM/x86 binaries for hardware trojans and backdoors.

YOUR CRITICAL TASK: Analyze disassembled functions to identify trojans, backdoors, and malicious behavior patterns.

MANDATORY REQUIREMENTS - YOU MUST FOLLOW EXACTLY:
1. You MUST assign a risk_score between 0.0 (completely benign) and 1.0 (definitely malicious)
2. You MUST identify specific malicious patterns if present
3. You MUST cite specific addresses/instructions as evidence
4. You MUST respond with VALID JSON ONLY - no markdown, no explanations before/after the JSON

TROJAN INDICATORS TO DETECT:
- Hardcoded credentials (passwords, keys, magic strings like "admin123", "DEBUG_KEY")
- Magic value triggers (0xdeadbeef, 0xcafebabe triggering special behavior)
- Authentication bypasses (strcmp against hardcoded strings)
- Anti-debugging (rdtsc timing, ptrace checks, VM detection via VMWARE/VBOX envvars)
- Privilege escalation (direct syscalls: setuid, setgid, execve)
- Covert C2 (socket creation + hardcoded IP addresses)
- Time bombs (date/time checks triggering payloads)

SCORING GUIDELINES:
- 0.0-0.2: Completely benign (logging, memory allocation, checksums)
- 0.3-0.5: Suspicious but inconclusive
- 0.6-0.8: Likely malicious (clear trojan indicators)
- 0.9-1.0: Definitely malicious (multiple trojan indicators)"""

    user_prompt = f"""ANALYZE THIS FUNCTION FOR TROJANS/BACKDOORS:

Binary: {binary_name}
Function: {function_name}

{disassembly}

RESPOND WITH JSON ONLY:
{{
  "function_name": "{function_name}",
  "risk_score": <0.0-1.0>,
  "is_malicious": <bool>,
  "categories": ["backdoor"|"anti-analysis"|"c2"|"privesc"|"exfil"|"benign"],
  "findings": [{{"type": "...", "description": "...", "evidence": "..."}}],
  "summary": "<one line>"
}}"""

    try:
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "reasoning": {"enabled": True},  # Enable thinking for MiMo
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            timeout=60,
        )
        
        response.raise_for_status()
        data = response.json()
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        # Parse JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(content[start:end])
            result["function_name"] = function_name
            return result
        
        return {
            "function_name": function_name,
            "risk_score": 0.0,
            "error": "JSON parse failed",
            "raw": content[:200]
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "function_name": function_name,
            "risk_score": 0.0,
            "error": f"API error: {str(e)}"
        }
    except Exception as e:
        return {
            "function_name": function_name,
            "risk_score": 0.0,
            "error": str(e)
        }


def analyze_binary(
    api_key: str,
    model: str,
    binary_data: Dict,
    rate_limit_delay: float = 1.0,
) -> Dict:
    """Analyze all functions in a binary."""
    functions = binary_data["functions"]
    ground_truth = set(binary_data.get("ground_truth_trojan_functions", []))
    
    results = []
    for i, fn in enumerate(functions):
        print(f"    [{i+1}/{len(functions)}] Analyzing {fn['name']}...", end=" ", flush=True)
        
        result = analyze_function(
            api_key=api_key,
            model=model,
            function_name=fn["name"],
            disassembly=fn["disassembly"],
            binary_name=binary_data["binary_id"],
        )
        
        results.append(result)
        
        risk = result.get("risk_score", 0)
        gt = "✓ TROJAN" if fn["name"] in ground_truth else ""
        print(f"risk={risk:.2f} {gt}")
        
        time.sleep(rate_limit_delay)
    
    # Sort by risk score (descending)
    ranked = sorted(results, key=lambda x: float(x.get("risk_score", 0)), reverse=True)
    
    # Compute metrics
    trojan_ranks = []
    for gt_name in ground_truth:
        for rank, fn in enumerate(ranked, 1):
            base_name = fn["function_name"].rsplit("_", 1)[0]
            gt_base = gt_name.rsplit("_", 1)[0]
            if base_name == gt_base:
                trojan_ranks.append(rank)
                break
    
    n_functions = len(functions)
    
    return {
        "binary_id": binary_data["binary_id"],
        "n_functions": n_functions,
        "ground_truth_trojans": list(ground_truth),
        "llm_rankings": [(r["function_name"], r.get("risk_score", 0)) for r in ranked[:10]],
        "trojan_ranks": trojan_ranks,
        "best_rank": min(trojan_ranks) if trojan_ranks else n_functions,
        "reciprocal_rank": 1.0 / min(trojan_ranks) if trojan_ranks else 0,
        "random_expected_rank": n_functions / 2,
        "full_results": results,
    }


def run_llm_analysis(
    binaries_dir: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_binaries: int = 10,
) -> List[Dict]:
    """Run LLM analysis on all synthetic binaries."""
    
    print(f"Using model: {model}")
    print(f"API: OpenRouter")
    
    # Load binaries
    binary_files = sorted(binaries_dir.glob("*.json"))[:max_binaries]
    print(f"Found {len(binary_files)} binary files")
    
    results = []
    for bf in binary_files:
        print(f"\n{'='*60}")
        print(f"Analyzing: {bf.name}")
        print(f"{'='*60}")
        
        with open(bf) as f:
            binary_data = json.load(f)
        
        result = analyze_binary(api_key, model, binary_data)
        results.append(result)
        
        print(f"\nResult: Best trojan rank = {result['best_rank']}/{result['n_functions']}")
        print(f"        MRR contribution = {result['reciprocal_rank']:.3f}")
    
    return results


def compute_aggregate(results: List[Dict]) -> Dict:
    """Compute aggregate metrics."""
    if not results:
        return {}
    
    mrr = np.mean([r["reciprocal_rank"] for r in results])
    avg_rank = np.mean([r["best_rank"] for r in results])
    avg_random = np.mean([r["random_expected_rank"] for r in results])
    improvement = avg_random / avg_rank if avg_rank > 0 else 0
    
    top3 = sum(1 for r in results if r["best_rank"] <= 3) / len(results)
    top5 = sum(1 for r in results if r["best_rank"] <= 5) / len(results)
    
    work_saved = []
    for r in results:
        saved = 1.0 - (r["best_rank"] / r["n_functions"])
        work_saved.append(saved)
    
    return {
        "n_binaries": len(results),
        "mrr": round(mrr, 3),
        "top3_rate": round(top3, 3),
        "top5_rate": round(top5, 3),
        "avg_llm_rank": round(avg_rank, 1),
        "avg_random_rank": round(avg_random, 1),
        "improvement_factor": round(improvement, 2),
        "work_saved_pct": round(np.mean(work_saved) * 100, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze synthetic binaries with OpenRouter LLM")
    ap.add_argument("--binaries-dir", type=Path, default=Path("data/synthetic_binaries"),
                    help="Directory with synthetic binary JSON files")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="OpenRouter model to use")
    ap.add_argument("--max-binaries", type=int, default=10)
    ap.add_argument("--output", type=Path, default=Path("results/llm_real/results.json"))
    args = ap.parse_args()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Set with: export OPENROUTER_API_KEY=your_key")
        return 1
    
    print("="*60)
    print("LLM ANALYSIS WITH OPENROUTER (MiMo-V2-Flash)")
    print("="*60)
    
    # Generate binaries if needed
    if not args.binaries_dir.exists() or not list(args.binaries_dir.glob("*.json")):
        print("Generating synthetic binaries first...")
        import subprocess
        subprocess.run([sys.executable, "experiments/generate_synthetic_binaries.py"], 
                      cwd=REPO_ROOT)
    
    results = run_llm_analysis(
        binaries_dir=args.binaries_dir,
        api_key=api_key,
        model=args.model,
        max_binaries=args.max_binaries,
    )
    
    metrics = compute_aggregate(results)
    
    print("\n" + "="*60)
    print("AGGREGATE METRICS")
    print("="*60)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "experiment": "llm_openrouter_analysis",
            "model": args.model,
            "metrics": metrics,
            "per_binary": results,
        }, f, indent=2)
    
    print(f"\nSaved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
