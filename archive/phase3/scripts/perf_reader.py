from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PerfSample:
    t: float
    values: Dict[str, float]


def parse_perf_stat_line(line: str) -> Optional[PerfSample]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None

    try:
        t = float(parts[0])
    except ValueError:
        return None

    raw_val = parts[1]
    event = parts[3]
    if not event:
        return None

    if raw_val in {"<not counted>", "<not supported>", ""}:
        return PerfSample(t=t, values={event.replace("-", "_"): float("nan")})

    raw_val = raw_val.replace(" ", "")
    try:
        val = float(raw_val.replace(",", ""))
    except ValueError:
        return None

    ev_key = event.replace("-", "_")
    return PerfSample(t=t, values={ev_key: val})
