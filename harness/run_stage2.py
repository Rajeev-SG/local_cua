"""Stage 2: three tiny executable tasks, 2 reps each, no retrying.

Per model role:
  - autonomous models (uitars/tongui/fara/showui-direct): model chooses each action
  - actor models (showui-actor, uground): gold next intent supplied by a scripted
    planner; the model only produces the low-level action at that step.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from . import tasks
from .browser import Executor
from .metrics import peak_rss_self_mb, score_point
from .adapters import get_adapter
from .run_stage1 import JSONL as S1

RESULTS = Path(__file__).resolve().parent.parent / "results"
JSONL = RESULTS / "stage2.jsonl"

# gold planner: yields the next subgoal + the DOM selector it refers to.
GOLD_PLANS = {
    "T1": [("Click the 'Open settings' button", "#open-settings")],
    "T2": [("Click the 'What needs to be done?' text field", "#new-todo"),
           ("Type 'Buy milk'", "#new-todo"),
           ("Click the 'Add' button", "#add-todo")],
    "T3": [("Click the text field 'What needs to be done?'", "#new-todo"),
           ("Type 'Buy milk'", "#new-todo"),
           ("Click the 'Add' button", "#add-todo"),
           ("Click the text field 'What needs to be done?'", "#new-todo"),
           ("Type 'Walk dog'", "#new-todo"),
           ("Click the 'Add' button", "#add-todo"),
           ("Click the checkbox next to 'Walk dog'", "li:last-child input[type=checkbox]"),
           ("Click the 'Active' filter tab", "#f-active")],
}


def _append(row):
    with JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")


def run(model_key, mode=None, reps=2, only=None):
    ex = Executor()
    ex.start()
    viewport = ex.viewport
    t0 = time.perf_counter()
    ad = get_adapter(model_key, viewport)
    ad.load()
    cold = time.perf_counter() - t0
    print(f"[{model_key}/{mode}] cold {cold:.1f}s peakRSS {peak_rss_self_mb()}MB")

    task_keys = [only] if only else ["T1", "T2", "T3"]
    for tk in task_keys:
        spec = tasks.TASKS[tk]
        for rep in range(reps):
            row = {"stage": 2, "model": ad.name, "family": ad.family, "role": ad.role,
                   "artifact": ad.artifact, "runtime": ad.runtime, "quant": ad.quant,
                   "mode": mode, "task": tk, "rep": rep + 1,
                   "cold_load_s": round(cold, 3), "local": ad.runtime != "cloud",
                   "task_instruction": spec.instruction}
            steps = []
            try:
                tasks.reset(ex)
                history = []
                passed = False
                e2e_times = []
                for step in range(spec.max_steps):
                    shot = str(RESULTS / "runs" / f"{model_key}_{mode}_{tk}_{rep}_{step}.png")
                    ex.screenshot(shot)
                    if mode == "actor" or model_key == "uground":
                        plan = GOLD_PLANS[tk]
                        if step >= len(plan):
                            break
                        subgoal, sel = plan[step]
                        bbox = ex.bbox(sel)
                        t_start = time.perf_counter()
                        pred = _actor_predict(ad, model_key, mode, shot, subgoal,
                                              history, viewport)
                        e2e_times.append((time.perf_counter() - t_start) * 1000)
                        # actor: use the model's point but the scripted action type
                        act = _scripted_action(ad, model_key, mode, shot, subgoal, pred, sel)
                        para = {"step": step, "subgoal": subgoal, "selector": sel,
                                "target_bbox": bbox, "native_text": pred.native_text,
                                "parse_ok": pred.parse_ok, "inference_ms": pred.inference_ms,
                                "actor_action": act}
                        if pred.point and bbox:
                            sc = score_point(pred.point[0], pred.point[1], bbox)
                            para.update({"predicted_point": list(pred.point),
                                         "hit": sc.hit})
                    else:
                        t_start = time.perf_counter()
                        pred = ad.predict(shot, spec.instruction, history=history)
                        e2e_times.append((time.perf_counter() - t_start) * 1000)
                        act = pred.action
                        para = {"step": step, "native_text": pred.native_text,
                                "parse_ok": pred.parse_ok, "inference_ms": pred.inference_ms,
                                "action_kind": act.kind,
                                "predicted_point": [act.x, act.y] if act.x is not None else None}
                    parsed_act = _execute(ex, act)
                    para["executor"] = parsed_act
                    history.append(("assistant", pred.native_text))
                    steps.append(para)
                    if act.kind in ("done", "answer"):
                        break
                passed = tasks.check(ex, spec)
                row.update({"steps": len(steps), "pass": passed,
                            "final_state": tasks.state(ex),
                            "step_detail": steps,
                            "e2e_ms": round(sum(e2e_times), 1) if e2e_times else None})
            except Exception as e:
                row.update({"pass": False, "error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-800:], "step_detail": steps})
            row["peak_rss_mb"] = peak_rss_self_mb()
        row["device_peak"] = device_peak_mb()
            _append(row)
            print(f"  {tk} rep{rep+1}: pass={row.get('pass')} steps={row.get('steps')}"
                  f"{' ERR=' + row.get('error','') if row.get('error') else ''}")
    ad.close()
    ex.stop()


def _load_state():
    return None


def _actor_predict(ad, model_key, mode, shot, subgoal, history, viewport):
    if model_key == "uground":
        return ad.predict(shot, subgoal, history=history)
    return ad.predict(shot, subgoal, history=history, mode=mode)


def _scripted_action(ad, model_key, mode, shot, subgoal, pred, sel):
    """Actor mode: the planner supplies the action *type*; the grounder only
    supplies the point. Returns a normalised action."""
    from .types import NormalisedAction
    low = subgoal.lower()
    if low.startswith("type "):
        text = subgoal.split("'")[1] if "'" in subgoal else subgoal[5:].strip()
        return NormalisedAction(kind="type", text=text)
    if pred.point and pred.action.x is not None:
        return NormalisedAction(kind="click", x=pred.action.x, y=pred.action.y)
    return NormalisedAction(kind="none")


def _execute(ex: Executor, act):
    from .types import NormalisedAction
    if act is None or act.kind in (None, "none", "done", "answer"):
        return "no-op"
    try:
        if act.kind == "click":
            r = ex.click(act.x, act.y)
        elif act.kind == "double_click":
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
            r = ex.navigate(act.url)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    run(a.model, a.mode, a.reps, a.only)
