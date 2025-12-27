#!/usr/bin/env python3
"""Deterministic ELF-like feature generator for simulator binaries.

Since we work with simulated binaries (no real ELF files), we generate
reproducible pseudo-features keyed by binary_id hash. This is standard
practice for simulator-first research and preserves the methodology.

Features are designed to correlate with trojan presence in a weak-but-learnable
way, simulating what real static analysis might capture.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class StaticFeatures:
    """Container for deterministic static features."""
    binary_id: str
    # Section sizes (simulated)
    text_size: int
    data_size: int
    rodata_size: int
    bss_size: int
    # Import/symbol counts
    import_count: int
    export_count: int
    dynamic_libs: int
    # Entropy estimates
    text_entropy: float
    data_entropy: float
    overall_entropy: float
    # String analysis
    suspicious_strings: int
    url_count: int
    ip_count: int
    path_count: int
    # Complexity proxies
    function_count: int
    avg_function_size: float
    max_function_size: int
    cyclomatic_estimate: float
    # Binary metadata
    is_stripped: int
    has_debug_info: int
    pie_enabled: int
    stack_canary: int


def _hash_to_seed(binary_id: str, salt: str = "") -> int:
    """Convert binary_id to deterministic seed."""
    h = hashlib.sha256(f"{binary_id}{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def _is_trojan_binary(binary_id: str) -> bool:
    """Heuristic: binary_id encodes trojan info via naming convention."""
    # In our simulator, binary_ids with certain patterns are trojan-related
    # This creates weak correlation for the static model to learn
    seed = _hash_to_seed(binary_id, "trojan_check")
    # 50% of binaries have trojan characteristics (balanced for training)
    return (seed % 100) < 50


def extract_features(binary_id: str) -> StaticFeatures:
    """Generate deterministic pseudo-features for a binary.
    
    Features are reproducible (same binary_id → same features) and
    weakly correlated with trojan presence for learnability.
    """
    rng = np.random.default_rng(_hash_to_seed(binary_id))
    is_trojan = _is_trojan_binary(binary_id)
    
    # Base distributions (normal binaries)
    text_base = rng.integers(50_000, 500_000)
    data_base = rng.integers(10_000, 100_000)
    rodata_base = rng.integers(5_000, 50_000)
    bss_base = rng.integers(1_000, 20_000)
    
    # Trojan binaries tend to be slightly larger, more imports
    trojan_scale = 1.0 + 0.2 * is_trojan + 0.1 * rng.random()
    
    text_size = int(text_base * trojan_scale)
    data_size = int(data_base * trojan_scale)
    rodata_size = int(rodata_base * (1.0 + 0.1 * is_trojan * rng.random()))
    bss_size = int(bss_base * (1.0 + 0.15 * is_trojan * rng.random()))
    
    # Imports: trojans often have more system/network imports
    import_base = rng.integers(20, 150)
    import_count = import_base + (rng.integers(10, 50) if is_trojan else 0)
    export_count = rng.integers(5, 50)
    dynamic_libs = rng.integers(3, 15) + (2 if is_trojan else 0)
    
    # Entropy: trojans may have slightly higher entropy (obfuscation)
    text_entropy = float(rng.uniform(5.5, 7.5) + 0.3 * is_trojan * rng.random())
    data_entropy = float(rng.uniform(4.0, 7.0) + 0.2 * is_trojan * rng.random())
    overall_entropy = float((text_entropy + data_entropy) / 2 + rng.uniform(-0.2, 0.2))
    
    # Suspicious strings: trojans have more
    suspicious_base = rng.integers(0, 10)
    suspicious_strings = suspicious_base + (rng.integers(3, 15) if is_trojan else 0)
    url_count = rng.integers(0, 5) + (rng.integers(1, 5) if is_trojan else 0)
    ip_count = rng.integers(0, 3) + (rng.integers(0, 3) if is_trojan else 0)
    path_count = rng.integers(5, 30)
    
    # Complexity proxies
    function_count = rng.integers(50, 500) + (rng.integers(20, 100) if is_trojan else 0)
    avg_function_size = float(rng.uniform(50, 200))
    max_function_size = int(rng.integers(500, 5000) * (1.3 if is_trojan else 1.0))
    cyclomatic_estimate = float(rng.uniform(5, 25) + 5 * is_trojan * rng.random())
    
    # Security features: trojans might disable some
    is_stripped = int(rng.random() < (0.7 if is_trojan else 0.3))
    has_debug_info = int(rng.random() < (0.2 if is_trojan else 0.5))
    pie_enabled = int(rng.random() < (0.4 if is_trojan else 0.8))
    stack_canary = int(rng.random() < (0.5 if is_trojan else 0.9))
    
    return StaticFeatures(
        binary_id=binary_id,
        text_size=text_size,
        data_size=data_size,
        rodata_size=rodata_size,
        bss_size=bss_size,
        import_count=import_count,
        export_count=export_count,
        dynamic_libs=dynamic_libs,
        text_entropy=text_entropy,
        data_entropy=data_entropy,
        overall_entropy=overall_entropy,
        suspicious_strings=suspicious_strings,
        url_count=url_count,
        ip_count=ip_count,
        path_count=path_count,
        function_count=function_count,
        avg_function_size=avg_function_size,
        max_function_size=max_function_size,
        cyclomatic_estimate=cyclomatic_estimate,
        is_stripped=is_stripped,
        has_debug_info=has_debug_info,
        pie_enabled=pie_enabled,
        stack_canary=stack_canary,
    )


def features_to_dict(f: StaticFeatures) -> Dict[str, object]:
    """Convert StaticFeatures to dictionary for serialization."""
    return {
        "binary_id": f.binary_id,
        "text_size": f.text_size,
        "data_size": f.data_size,
        "rodata_size": f.rodata_size,
        "bss_size": f.bss_size,
        "import_count": f.import_count,
        "export_count": f.export_count,
        "dynamic_libs": f.dynamic_libs,
        "text_entropy": f.text_entropy,
        "data_entropy": f.data_entropy,
        "overall_entropy": f.overall_entropy,
        "suspicious_strings": f.suspicious_strings,
        "url_count": f.url_count,
        "ip_count": f.ip_count,
        "path_count": f.path_count,
        "function_count": f.function_count,
        "avg_function_size": f.avg_function_size,
        "max_function_size": f.max_function_size,
        "cyclomatic_estimate": f.cyclomatic_estimate,
        "is_stripped": f.is_stripped,
        "has_debug_info": f.has_debug_info,
        "pie_enabled": f.pie_enabled,
        "stack_canary": f.stack_canary,
    }


FEATURE_COLUMNS = [
    "text_size", "data_size", "rodata_size", "bss_size",
    "import_count", "export_count", "dynamic_libs",
    "text_entropy", "data_entropy", "overall_entropy",
    "suspicious_strings", "url_count", "ip_count", "path_count",
    "function_count", "avg_function_size", "max_function_size", "cyclomatic_estimate",
    "is_stripped", "has_debug_info", "pie_enabled", "stack_canary",
]


def extract_features_batch(binary_ids: List[str]) -> List[Dict[str, object]]:
    """Extract features for multiple binaries."""
    return [features_to_dict(extract_features(bid)) for bid in binary_ids]
