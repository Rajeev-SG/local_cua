"""Stage 1: 8-action grounding screen.

For each scene: load fixture, screenshot, ask the adapter for one action/point,
score the predicted point against the DOM bbox, append a JSONL row.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from .browser import Executor, fixture_url
from .capture import SCENES, OUT as FIX_OUT
from .metrics import score_point, peak_rss_self_mb
from .adapters import get_adapter

RESULTS = Path(__file__).resolve().parent.parent / "results"
JSONL = RESULTS / "stage1.jsonl"


def ensure_fixtures() -> dict:
    idx = FIX_OUT / "index.json"
    if not idx.exists():
        from .capture import main as cap
        cap()
    return json.loads(idx.read_text())


def run(model_key: str, warm_reps: int = 3, limit: int | None = None) -> None:
    fixtures = ensure_fixtures()
    viewport = fixtures["scene1"]["viewport"]

    # Cold-load timing is captured inside the adapter.
    t0 = time.perf_counter()
    ad = get_adapter(model_key, viewport)
    ad.load()
    cold_load_s = time.perf_counter() - t0
    print(f"[{model_key}] cold load {cold_load_s:.1f}s, peak RSS {peak_rss_self_mb()} MB")

    RESULTS.mkdir(parents=True, exist_ok=True)
    items = list(fixtures.items())
    if limit:
        items = items[:limit]

    # warm-up on scene 1 (not scored)
    first = items[0][1]
    for _ in range(max(0, warm_reps - 1)):
        try:
            ad.predict(first["screenshot"], first["instruction"])
        except Exception:
            pass

    for name, meta in items:
        row = {
            "stage": 1,
            "model": ad.name,
            "family": ad.family,
            "role": ad.role,
            "artifact": ad.artifact,
            "runtime": ad.runtime,
            "quant": ad.quant,
            "scene": name,
            "instruction": meta["instruction"],
            "target_bbox": meta["bbox"],
            "cold_load_s": round(cold_load_s, 3),
            "local": ad.runtime != "cloud",
        }
        try:
            times = []
            pred = None
            for i in range(warm_reps):
                pred = ad.predict(meta["screenshot"], meta["instruction"])
                times.append(pred.inference_ms)
            times.sort()
            pred.inference_ms = times[len(times) // 2]  # warm p50
            row.update({
                "native_text": pred.native_text,
                "normalised_action": pred.action.__dict__,
                "parse_ok": pred.parse_ok,
                "parse_ms": round(pred.parse_ms, 3),
                "inference_ms": round(pred.inference_ms, 3),
                "inference_ms_samples": [round(t, 3) for t in times],
                "parse_error": pred.parse_error,
            })
            pt = pred.point or (pred.action.x, pred.action.y)
            if pt[0] is not None and pt[1] is not None and meta["bbox"]:
                sc = score_point(pt[0], pt[1], meta["bbox"])
                row.update({
                    "predicted_point": [pt[0], pt[1]],
                    "hit": sc.hit,
                    "dist_to_center_px": round(sc.dist_to_center_px, 1),
                    "dist_to_bbox_px": round(sc.dist_to_bbox_px, 1),
                })
            else:
                row.update({"predicted_point": None, "hit": False,
                            "malformed": True})
        except Exception as e:
            row.update({
                "hit": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-800:],
            })
        row["peak_rss_mb"] = peak_rss_self_mb()
        with JSONL.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {name}: hit={row.get('hit')} "
              f"infer={row.get('inference_ms')}ms "
              f"pt={row.get('predicted_point')}")

    ad.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--warm-reps", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model, a.warm_reps, a.limit)
