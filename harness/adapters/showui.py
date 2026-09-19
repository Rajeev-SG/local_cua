"""ShowUI-2B adapter (mlx-vlm) — native web action schema.

Official contracts (showlab/ShowUI):
  Grounding  : system "Based on the screenshot of the page, I give a text
               description and you give its corresponding location..." ->
               output is a relative [x, y] list.
  Navigation : _NAV_SYSTEM + _NAV_FORMAT with a web action space ->
               output is {'action', 'value', 'position'} dicts, with
               *relative* 0-1 positions.

Two modes (issue #4):
  direct -> full task instruction given to the model (navigation prompt)
  actor  -> the next subgoal/intent is supplied; model only grounds it
"""
from __future__ import annotations

import ast
import json
import re
import time

from ..types import Adapter, NormalisedAction, Prediction, template_prompt
from . import register

GROUNDING_SYSTEM = ("Based on the screenshot of the page, I give a text description "
                    "and you give its corresponding location. The coordinate represents "
                    "a clickable location [x, y] for an element, which is a relative "
                    "coordinate on the screenshot, scaled from 0 to 1.")

_ACTION_SPACE_WEB = """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required.
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required.
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required.
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
"""

_NAV_SYSTEM = f"""You are an assistant trained to navigate the web screen.
Given a task instruction, a screen observation, and an action history sequence,
output the next action and wait for the next observation.
Here is the action space:
{_ACTION_SPACE_WEB}
"""

_NAV_FORMAT = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

WEB_ACTIONS = {"CLICK", "INPUT", "SELECT", "HOVER", "ANSWER", "ENTER",
               "SCROLL", "SELECT_TEXT", "COPY"}


@register("showui")
class ShowUIAdapter(Adapter):
    name = "ShowUI-2B"
    family = "qwen2-vl"
    role = "actor+agent"
    artifact = "mlx-community/ShowUI-2B-bf16-4bit"
    runtime = "mlx-vlm"
    quant = "bf16-4bit"

    def load(self):
        from mlx_vlm import load
        self.model, self.processor = load(self.artifact)

    def predict(self, image_path, instruction, history=None, mode="direct"):
        from mlx_vlm import generate
        if mode == "actor":
            # subgoal supplied: ground-only prompt
            prompt = f"{GROUNDING_SYSTEM}\n\n{instruction}"
        else:
            hist = ""
            if history:
                hist = "\nPast actions:\n" + "\n".join(
                    f"- {a}" for _, a in history[-4:])
            prompt = f"{_NAV_SYSTEM}{_NAV_FORMAT}\nTask: {instruction}{hist}"

        prompt = template_prompt(self.processor, self.artifact, prompt)
        t0 = time.perf_counter()
        res = generate(self.model, self.processor, prompt, image=image_path,
                       max_tokens=160, temperature=0.0)
        infer_ms = (time.perf_counter() - t0) * 1000
        text = res.text if hasattr(res, "text") else str(res)

        t1 = time.perf_counter()
        action, err = parse_showui(text, self.viewport, mode)
        parse_ms = (time.perf_counter() - t1) * 1000
        return Prediction(
            model=self.name, native_text=text, action=action,
            parse_ok=err is None, parse_ms=parse_ms, inference_ms=infer_ms,
            point=(action.x, action.y) if action.x is not None else None,
            parse_error=err, extra={"mode": mode},
        )


def _rel(v, extent):
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if 0.0 <= f <= 1.0:
        return f * extent
    if f <= 1000.0:
        return f / 1000.0 * extent
    return f


def parse_showui(text: str, viewport: dict, mode: str = "direct"):
    """Preserve the native structure; normalise position -> viewport px."""
    # 1) navigation dict form
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0).replace("'", '"'))
        except Exception:
            try:
                d = ast.literal_eval(m.group(0))
            except Exception:
                d = None
        if isinstance(d, dict) and "action" in d:
            act = str(d.get("action", "")).upper()
            pos = d.get("position")
            x = y = None
            if isinstance(pos, (list, tuple)) and len(pos) == 2 and \
                    all(isinstance(p, (int, float)) for p in pos):
                x = _rel(pos[0], viewport["width"])
                y = _rel(pos[1], viewport["height"])
            elif isinstance(pos, (list, tuple)) and len(pos) == 2:
                # drag/select: [[x1,y1],[x2,y2]]
                (x1, y1), (x2, y2) = pos
                x = _rel(x1, viewport["width"]); y = _rel(y1, viewport["height"])
            kind_map = {"CLICK": "click", "INPUT": "type", "SELECT": "click",
                        "HOVER": "move", "ANSWER": "answer", "ENTER": "key",
                        "SCROLL": "scroll", "SELECT_TEXT": "click", "COPY": "key"}
            na = NormalisedAction(kind=kind_map.get(act, "none"), x=x, y=y, raw=d)
            if act == "INPUT":
                na.text = str(d.get("value", ""))
            if act == "ENTER":
                na.keys = "Enter"
            if act == "COPY":
                na.keys = "Meta+C"
            if act == "SCROLL":
                na.dy = 500 if str(d.get("value", "down")).lower() == "down" else -500
            if act == "ANSWER":
                na.text = str(d.get("value", ""))
            err = None if act in WEB_ACTIONS else f"unknown action {act!r}"
            return na, err
    # 2) grounding list form [x, y]
    lm = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text)
    if lm:
        x = _rel(lm.group(1), viewport["width"])
        y = _rel(lm.group(2), viewport["height"])
        return (NormalisedAction(kind="click", x=x, y=y, raw={"list": text}),
                None if mode == "actor" else "list form in direct mode")
    return (NormalisedAction(kind="none", raw={"native": text}),
            "no action parsed")
