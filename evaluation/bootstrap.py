from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float


def bootstrap_over_runs(
    run_ids: Sequence[str],
    values: Sequence[float],
    *,
    n_boot: int = 500,
    seed: int = 1337,
) -> BootstrapCI:
    """Compute a percentile bootstrap CI over run-level scalar values."""
    assert len(run_ids) == len(values)
    run_ids = list(run_ids)
    values = np.asarray(values, dtype=np.float64)
    uniq = sorted(set(run_ids))
    if not uniq:
        return BootstrapCI(mean=float("nan"), lo=float("nan"), hi=float("nan"))

    # map run -> mean value for that run
    per_run: Dict[str, float] = {}
    for r in uniq:
        idx = [i for i, rr in enumerate(run_ids) if rr == r]
        per_run[r] = float(np.nanmean(values[idx]))

    rng = np.random.default_rng(seed)
    samples: List[float] = []
    for _ in range(int(n_boot)):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        samples.append(float(np.nanmean([per_run[d] for d in draw])))

    samples_np = np.asarray(samples, dtype=np.float64)
    return BootstrapCI(
        mean=float(np.nanmean(list(per_run.values()))),
        lo=float(np.nanpercentile(samples_np, 2.5)),
        hi=float(np.nanpercentile(samples_np, 97.5)),
    )
