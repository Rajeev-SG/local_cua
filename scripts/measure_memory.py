"""Measure resident + peak accelerator memory for one model, on this machine.

Usage: .venv/bin/python scripts/measure_memory.py <adapter-key>
Prints JSON: {resident_gb, peak_gb, rss_gb}.
Resident = memory held after one real inference. Peak = high-water mark.
On Apple Silicon these live in unified memory and do NOT show up in RSS.
"""
import json
import subprocess
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SHOT = "results/fixtures/stage1/scene1/shot.png"
PROMPT = "Click the button"


def rss_gb():
    return round(int(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1048576, 2)


def main(key):
    from harness.adapters import get_adapter
    ad = get_adapter(key, {"width": 1440, "height": 900})
    ad.load()
    ad.predict(SHOT, PROMPT)
    out = {"model": ad.name, "artifact": ad.artifact, "quant": ad.quant,
           "runtime": ad.runtime, "rss_gb": rss_gb()}
    try:
        import mlx.core as mx
        out["resident_gb"] = round(mx.get_active_memory() / 1e9, 2)
        out["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            out["resident_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 2)
            out["peak_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    except Exception:
        pass
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "showui")
