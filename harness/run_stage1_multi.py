"""Stage 1 variant for adapters that need a mode argument (ShowUI actor/direct)."""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
from .capture import OUT as FIX_OUT
from .metrics import score_point, peak_rss_self_mb, device_peak_mb
from .adapters import get_adapter

RESULTS = Path(__file__).resolve().parent.parent / "results"
JSONL = RESULTS / "stage1.jsonl"


def run(model_key, mode, warm_reps=3):
    fixtures = json.loads((FIX_OUT / "index.json").read_text())
    viewport = fixtures["scene1"]["viewport"]
    t0 = time.perf_counter()
    ad = get_adapter(model_key, viewport)
    ad.load()
    cold = time.perf_counter() - t0
    print(f"[{model_key}/{mode}] cold {cold:.1f}s RSS {peak_rss_self_mb()}MB")
    first = fixtures["scene1"]
    # For the actor mode, supply the gold element description as the subgoal.
    for _ in range(max(0, warm_reps - 1)):
        try:
            ad.predict(first["screenshot"], first["instruction"], mode=mode)
        except Exception:
            pass
    for name, meta in fixtures.items():
        row = {"stage": 1, "model": ad.name, "family": ad.family, "role": ad.role,
               "artifact": ad.artifact, "runtime": ad.runtime, "quant": ad.quant,
               "mode": mode, "scene": name, "instruction": meta["instruction"],
               "target_bbox": meta["bbox"], "cold_load_s": round(cold, 3),
               "local": ad.runtime != "cloud"}
        try:
            times = []
            pred = None
            for i in range(warm_reps):
                pred = ad.predict(meta["screenshot"], meta["instruction"], mode=mode)
                times.append(pred.inference_ms)
            times.sort()
            pred.inference_ms = times[len(times) // 2]
            row.update({"native_text": pred.native_text,
                        "normalised_action": pred.action.__dict__,
                        "parse_ok": pred.parse_ok, "parse_ms": round(pred.parse_ms, 3),
                        "inference_ms": round(pred.inference_ms, 3),
                        "inference_ms_samples": [round(t, 3) for t in times],
                        "parse_error": pred.parse_error})
            pt = pred.point or (pred.action.x, pred.action.y)
            if pt[0] is not None and meta["bbox"]:
                sc = score_point(pt[0], pt[1], meta["bbox"])
                row.update({"predicted_point": [pt[0], pt[1]], "hit": sc.hit,
                            "dist_to_center_px": round(sc.dist_to_center_px, 1)})
            else:
                row.update({"predicted_point": None, "hit": False, "malformed": True})
        except Exception as e:
            row.update({"hit": False, "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-600:]})
        row["peak_rss_mb"] = peak_rss_self_mb()
        row["device_peak"] = device_peak_mb()
        with JSONL.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {name}: hit={row.get('hit')} infer={row.get('inference_ms')}ms pt={row.get('predicted_point')}")
    ad.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--mode", default="direct")
    ap.add_argument("--warm-reps", type=int, default=3)
    a = ap.parse_args()
    run(a.model, a.mode, a.warm_reps)
