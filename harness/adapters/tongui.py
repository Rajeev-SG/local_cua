"""TongUI-3B adapter (Transformers on MPS).

Native contract (TongUI-agent repo, tongui/data/template/navigation_prompt_v2.py):
  - Navigation prompt _NAV_SYSTEM (desktop action space) + _NAV_FORMAT -> a
    "Thought: ... Action: <JSON or JSON array>" response.
  - Actions: CLICK, INPUT, SCROLL, LEFT_CLICK_DOUBLE, RIGHT_CLICK_SINGLE,
    DRAG, HOT_KEY, WAIT, FINISH. Position is *relative* 0-1.
  - May emit multiple actions per call (JSON array) - preserved, not forced
    into one.
  - Grounding prompt (examples/inference.py): system text asking for a
    clickable [x, y] relative coordinate.
"""
from __future__ import annotations

import ast
import json
import re
import time

from ..types import Adapter, NormalisedAction, Prediction
from . import register

GROUNDING_SYSTEM = ("Based on the screenshot of the page, I give a text description "
                    "and you give its corresponding location. The coordinate represents "
                    "a clickable location [x, y] for an element, which is a relative "
                    "coordinate on the screenshot, scaled from 0 to 1.")

ACTION_SPACE_DESKTOP = """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required.
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required.
3. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is start position of the scroll operation.
4. `LEFT_CLICK_DOUBLE`: Left click on an element twice, value is not applicable and the position [x,y] is required.
5. `RIGHT_CLICK_SINGLE`: Right click on an element once, value is not applicable and the position [x,y] is required.
6. `DRAG`: Drag the cursor to the specified position with the left button pressed. Value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the drag operation.
7. `HOT_KEY`: Press a hot key, value is the hot key and the position is not applicable.
8. `WAIT`: Wait for 5 seconds, and take a screenshot to check for any changes. Value and position are not applicable.
9. `FINISH`: Finish the task. Value and position are not applicable.
"""

_NAV_SYSTEM = f"""You are an assistant trained to navigate the computer screen.
Given a task instruction, a screen observation, and an action history sequence,
output the next action and wait for the next observation.
Here is the action space:
{ACTION_SPACE_DESKTOP}
"""

_NAV_FORMAT = """
Format your response as
Thought: <your reasoning process>
Action: <the next action>

Format the action as a JSON object with the following keys:
{"action": "ACTION_TYPE", "value": "element", "position": [x,y]}

You can output multiple actions at once, and use JSON array to represent multiple actions.
If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""


@register("tongui")
class TongUIAdapter(Adapter):
    name = "TongUI-3B"
    family = "qwen2.5-vl"
    role = "agent"
    artifact = "Bofeee5675/TongUI-3B"
    runtime = "transformers-mps"
    quant = "bf16"

    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            self.artifact, min_pixels=256 * 28 * 28, max_pixels=1344 * 28 * 28,
            model_max_length=8196)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.artifact, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.model.to("mps")
        self.model.eval()

    def predict(self, image_path, instruction, history=None):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        hist = ""
        if history:
            hist = "\nAction History: " + "; ".join(a for _, a in history[-4:])
        body = f"{_NAV_SYSTEM}{_NAV_FORMAT}\nTask: {instruction}{hist}\nWhat is the next action?"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": body},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: (v.to("mps") if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        infer_ms = (time.perf_counter() - t0) * 1000
        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(gen, skip_special_tokens=True)[0]

        t1 = time.perf_counter()
        actions, err = parse_tongui(raw, self.viewport)
        parse_ms = (time.perf_counter() - t1) * 1000
        first = actions[0] if actions else NormalisedAction(kind="none")
        return Prediction(
            model=self.name, native_text=raw, action=first,
            parse_ok=err is None, parse_ms=parse_ms, inference_ms=infer_ms,
            point=(first.x, first.y) if first.x is not None else None,
            parse_error=err,
            extra={"n_actions": len(actions),
                   "all_actions": [_serial(a) for a in actions]},
        )


def _serial(a: NormalisedAction) -> dict:
    return {"kind": a.kind, "x": a.x, "y": a.y, "text": a.text, "keys": a.keys,
            "dy": a.dy}


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


KIND_MAP = {
    "CLICK": "click", "INPUT": "type", "SCROLL": "scroll",
    "LEFT_CLICK_DOUBLE": "double_click", "RIGHT_CLICK_SINGLE": "right_click",
    "DRAG": "drag", "HOT_KEY": "key", "WAIT": "wait", "FINISH": "done",
    "SELECT": "click", "HOVER": "move", "ANSWER": "answer",
    "ENTER": "key", "SELECT_TEXT": "click", "COPY": "key",
}


def parse_tongui(text: str, viewport: dict):
    """Parse TongUI JSON action(s) (relative 0-1 positions)."""
    # strip a Thought: prefix; find JSON object(s) or array
    objs = []
    arr = re.search(r"\[(\s*\{.*?\}\s*,?\s*)+\]", text, re.S)
    if arr:
        try:
            parsed = json.loads(arr.group(0))
            if isinstance(parsed, list):
                objs = parsed
        except Exception:
            objs = []
    if not objs:
        for m in re.finditer(r"\{[^{}]*\}", text, re.S):
            try:
                objs.append(json.loads(m.group(0)))
            except Exception:
                try:
                    objs.append(ast.literal_eval(m.group(0)))
                except Exception:
                    continue
    if not objs:
        return ([], "no JSON action found")
    out = []
    for d in objs:
        if not isinstance(d, dict):
            continue
        act = str(d.get("action", "")).upper()
        pos = d.get("position")
        x = y = dx2 = dy2 = None
        if isinstance(pos, (list, tuple)) and len(pos) == 2 and \
                all(isinstance(p, (int, float, type(None))) for p in pos):
            x = _rel(pos[0], viewport["width"]); y = _rel(pos[1], viewport["height"])
        elif isinstance(pos, (list, tuple)) and len(pos) == 2:
            (a1, b1), (a2, b2) = pos
            x = _rel(a1, viewport["width"]); y = _rel(b1, viewport["height"])
            dx2 = _rel(a2, viewport["width"]); dy2 = _rel(b2, viewport["height"])
        na = NormalisedAction(kind=KIND_MAP.get(act, "none"), x=x, y=y,
                              x2=dx2, y2=dy2, raw=d)
        val = d.get("value")
        if act == "INPUT":
            na.text = str(val) if val is not None else ""
        if act == "SCROLL":
            na.dy = 500 if str(val).lower() == "down" else -500
        if act == "HOT_KEY":
            na.keys = str(val) if val else "Enter"
        if act == "FINISH":
            na.text = str(val) if val is not None else ""
        out.append(na)
    err = None if out else "no parseable action"
    return (out, err)
