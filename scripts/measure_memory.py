"""Measure resident + peak accelerator memory for one model on this machine.

Usage: .venv/bin/python scripts/measure_memory.py <adapter-key> [--runs N]
Prints JSON with an explicit `backend` field and `error` if nothing measured.

Only the backend that actually runs the adapter is probed. torch and MLX are
both usually installed here, and on any Apple Silicon Mac
`torch.backends.mps.is_available()` is True regardless of which runtime is in
use - probing both and letting torch overwrite MLX once produced wrong MLX
numbers, so this dispatches on `ad.runtime` instead.

Resident = memory held after a WARM-UP inference (KV cache grown, lazy weights
materialised), reported as the median of `--runs` measurements. Peak = high-water
mark. On Apple Silicon these live in unified memory and do NOT appear in RSS.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SHOT = "results/fixtures/stage1/scene1/shot.png"
PROMPT = "Click the button"


def rss_gb() -> float:
    return round(int(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1048576, 2)


def is_mlx(runtime: str) -> bool:
    return "mlx" in (runtime or "").lower()


def is_torch(runtime: str) -> bool:
    r = (runtime or "").lower()
    return "mps" in r or "transformers" in r or "torch" in r


def probe_mlx():
    import mlx.core as mx
    return {"backend": "mlx",
            "resident_gb": round(mx.get_active_memory() / 1e9, 2),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}


def probe_torch():
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("torch MPS not available")
    return {"backend": "torch-mps",
            "resident_gb": round(torch.mps.current_allocated_memory() / 1e9, 2),
            "peak_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2)}


def main(key: str, runs: int) -> None:
    from harness.adapters import get_adapter
    ad = get_adapter(key, {"width": 1440, "height": 900})
    ad.load()
    ad.predict(SHOT, PROMPT)          # warm-up: grow KV cache, materialise weights

    if is_mlx(ad.runtime):
        probe = probe_mlx
    elif is_torch(ad.runtime):
        probe = probe_torch
    else:
        probe = None

    out = {"model": ad.name, "artifact": ad.artifact, "quant": ad.quant,
           "runtime": ad.runtime, "rss_gb": rss_gb(), "runs": runs}
    if probe is None:
        out["error"] = f"no memory probe for runtime {ad.runtime!r}"
        print(json.dumps(out, indent=2))
        raise SystemExit(2)

    residents, peaks, backend = [], [], None
    for _ in range(runs):
        ad.predict(SHOT, PROMPT)      # each measurement is post-warm-up
        m = probe()
        backend = m["backend"]
        residents.append(m["resident_gb"])
        peaks.append(m["peak_gb"])
    out.update({
        "backend": backend,
        "resident_gb": round(statistics.median(residents), 2),
        "resident_gb_min": min(residents), "resident_gb_max": max(residents),
        "peak_gb": round(max(peaks), 2),   # high-water mark, not resettable for torch
    })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?", default="showui")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    main(a.key, a.runs)
