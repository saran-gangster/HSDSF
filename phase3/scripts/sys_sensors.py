from pathlib import Path
from typing import Dict, List, Tuple


def discover_thermal_zones() -> List[Tuple[str, Path]]:
    zones: List[Tuple[str, Path]] = []
    base = Path("/sys/class/thermal")
    if not base.exists():
        return zones
    for zdir in base.glob("thermal_zone*"):
        tfile = zdir / "type"
        tempfile = zdir / "temp"
        if tfile.exists() and tempfile.exists():
            ztype = tfile.read_text().strip()
            zones.append((ztype, tempfile))
    return zones


def read_thermal_c(zones: List[Tuple[str, Path]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ztype, tempfile in zones:
        try:
            raw = tempfile.read_text().strip()
            val = float(raw)
        except Exception:
            continue
        c = val / 1000.0 if val > 200 else val
        key = f"temp_{ztype.lower()}_c".replace(" ", "_")
        out[key] = c
    return out
