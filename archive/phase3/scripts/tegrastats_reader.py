import re
from typing import Dict

RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
RE_CPU = re.compile(r"CPU\s+\[(.*?)\]")
RE_GPU = re.compile(r"GR3D_FREQ\s+(\d+)%")
RE_EMC = re.compile(r"EMC_FREQ\s+(\d+)%")
RE_TEMP = re.compile(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C")
RE_PWR = re.compile(r"(POM_[A-Za-z0-9_]+)\s+(\d+)mW")


def parse_tegrastats_line(line: str) -> Dict[str, float]:
    out: Dict[str, float] = {}

    m = RE_RAM.search(line)
    if m:
        out["ts_ram_used_mb"] = float(m.group(1))
        out["ts_ram_total_mb"] = float(m.group(2))

    m = RE_GPU.search(line)
    if m:
        out["ts_gpu_util"] = float(m.group(1))

    m = RE_EMC.search(line)
    if m:
        out["ts_emc_util"] = float(m.group(1))

    m = RE_CPU.search(line)
    if m:
        items = [s.strip() for s in m.group(1).split(",")]
        util_sum = 0.0
        for it in items:
            um = re.match(r"(\d+)%@(\d+)", it)
            if um:
                util_sum += float(um.group(1))
        out["ts_cpu_util_sum"] = util_sum

    for name, temp in RE_TEMP.findall(line):
        key = f"ts_temp_{name.lower()}_c"
        out[key] = float(temp)

    for rail, mw in RE_PWR.findall(line):
        key = f"ts_pwr_{rail.lower()}_mw"
        out[key] = float(mw)

    return out
