"""Stage 3: real-work delegation frontier over the harvested corpus.

Read-only over `web-automation-microbench/bench-ext/corpus/tasks/*.json`. Runs a
Fara-family adapter (whose native action space has NO JS eval) against live-site
tasks, then judges each run with the corpus's OWN verifier: `verification.verify_js`
recomputes ground truth in the page and `pass_rule.py` decides pass/fail against the
agent's `window.__bench_finding`. The adapter is never handed eval; to let an agent
report a finding, the harness relays the JSON object the model puts in a
`terminate`/`read_page_answer_question` answer into `__bench_finding`. That relay is
identical for every model, so the comparison is fair.

Live sites load slowly: navigation uses domcontentloaded with a long timeout.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
import traceback
from pathlib import Path

CORPUS = Path("/Users/rajeev/Code/web-automation-microbench/bench-ext")
sys.path.insert(0, str(CORPUS))
import pass_rule  # noqa: E402

from .browser import Executor  # noqa: E402
from .metrics import peak_rss_self_mb, device_peak_mb  # noqa: E402
from .adapters import get_adapter  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
JSONL = RESULTS / "stage3.jsonl"
RUNS = RESULTS / "runs"

# Stratified real-work set (>=5), chosen for class coverage, NOT to favour a model.
#   simple find/open/retrieve | filters/forms/report | short deterministic multi-step
#   longer/stateful | ambiguous
STRATA = {
    "porsche-uk-script-inventory": "1-simple-retrieve",
    "gymshark-uk-add-to-cart-tag-check": "2-form-commerce",
    "tldraw-three-shape-diagram": "3-short-multistep",
    "rajeevg-crawlability-audit": "4-longer-stateful",
    "puma-uk-seo-metadata-audit": "5-ambiguous-multifield",
    "puma-uk-tag-inspection": "2-form-commerce",
}

DIRECTIVE = (
    "\n\nWhen you have the final answer, end the run by emitting "
    '{"name":"computer_use","arguments":{"action":"terminate","answer":"<JSON>"}} '
    "where <JSON> is EXACTLY the JSON object the task asks you to record (no prose, no "
    "code fences). The answer field must be that raw JSON object."
)


def _extract_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _execute(ex, act):
    if act is None or act.kind in (None, "none", "done", "answer"):
        return "no-op"
    try:
        if act.kind == "click":
            r = ex.click(act.x, act.y)
        elif act.kind in ("double_click",):
            r = ex.double_click(act.x, act.y)
        elif act.kind == "right_click":
            r = ex.right_click(act.x, act.y)
        elif act.kind == "move":
            r = ex.move(act.x, act.y)
        elif act.kind == "type":
            r = ex.type_text(act.text or "")
        elif act.kind == "key":
            r = ex.key(act.keys or "Enter")
        elif act.kind == "scroll":
            r = ex.scroll(act.dx or 0, act.dy or 400)
        elif act.kind == "navigate" and act.url:
            r = ex.navigate(act.url, wait_until="domcontentloaded", timeout_ms=30000)
        elif act.kind == "back":
            r = ex.back()
        elif act.kind == "wait":
            time.sleep(1.0)
            return "wait"
        else:
            return f"unsupported:{act.kind}"
        ex.settle()
        return f"{r.detail} ({r.executor_ms:.0f}ms)"
    except Exception as e:
        return f"exec-error: {type(e).__name__}: {e}"


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def run(model_key, tasks, max_steps=14, nav_timeout=30000):
    ex = Executor()
    ex.start()
    viewport = ex.viewport
    t0 = time.perf_counter()
    ad = get_adapter(model_key, viewport)
    ad.load()
    cold = time.perf_counter() - t0
    print(f"[{model_key}] cold {cold:.1f}s peakRSS {peak_rss_self_mb()}MB", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    for tid in tasks:
        spec = json.loads((CORPUS / "corpus" / "tasks" / f"{tid}.json").read_text())
        instr = spec["instruction"] + DIRECTIVE
        row = {"stage": 3, "model": ad.name, "family": ad.family, "role": ad.role,
               "artifact": ad.artifact, "runtime": ad.runtime, "quant": ad.quant,
               "task": tid, "stratum": STRATA.get(tid, "?"), "url": spec["url"],
               "capabilities": spec.get("capabilities"), "pass_rule_kind":
               (spec["verification"].get("pass_rule") or {}).get("kind"),
               "cold_load_s": round(cold, 3), "local": ad.runtime != "cloud"}
        steps, lat, hist = [], [], []
        wall0 = time.perf_counter()
        try:
            ex.goto(spec["url"], wait_until="domcontentloaded", timeout_ms=nav_timeout)
            answered = False
            for step in range(max_steps):
                shot = str(RUNS / f"s3_{model_key}_{tid}_{step}.png")
                ex.screenshot(shot)
                pred = ad.predict(shot, instr, history=hist)
                lat.append(pred.inference_ms)
                act = pred.action
                para = {"step": step, "native_text": pred.native_text[:600],
                        "parse_ok": pred.parse_ok, "inference_ms": round(pred.inference_ms, 1),
                        "action_kind": act.kind, "text": (act.text or "")[:400]}
                para["executor"] = _execute(ex, act)
                hist.append(("assistant", pred.native_text))
                steps.append(para)
                if act.kind in ("done", "answer"):
                    obj = _extract_json(act.text)
                    if obj is not None:
                        ex.page.evaluate("(v) => { window.__bench_finding = v; }", obj)
                        answered = True
                    break
            raw = ex.page.evaluate(spec["verification"]["verify_js"])
            v = json.loads(raw) if isinstance(raw, str) else raw
            truth, finding = v.get("truth"), v.get("finding")
            passed = pass_rule.evaluate(spec["verification"].get("pass_rule"), truth, finding)
            fail = None
            if not passed:
                if not answered:
                    fail = "no-finding-emitted (action space has no eval/answer channel)"
                elif finding in (None, {}, ""):
                    fail = "finding-not-relayed"
                else:
                    fail = "finding-mismatch"
            row.update({"pass": bool(passed), "answered": answered, "steps": len(steps),
                        "truth": truth, "finding": finding, "failure_type": fail,
                        "step_detail": steps})
        except Exception as e:
            row.update({"pass": False, "steps": len(steps),
                        "failure_type": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-800:], "step_detail": steps})
        row["wall_time_s"] = round(time.perf_counter() - wall0, 2)
        row["turn_p50_ms"] = round(st.median(lat), 1) if lat else None
        row["turn_p95_ms"] = round(_pct(lat, 0.95), 1) if lat else None
        row["peak_rss_mb"] = peak_rss_self_mb()
        row["device_peak"] = device_peak_mb()
        with JSONL.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {tid}: pass={row.get('pass')} steps={row.get('steps')} "
              f"fail={row.get('failure_type')} wall={row['wall_time_s']}s", flush=True)

    ad.close()
    ex.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--max-steps", type=int, default=14)
    a = ap.parse_args()
    tks = a.tasks.split(",") if a.tasks else list(STRATA.keys())
    run(a.model, tks, a.max_steps)
