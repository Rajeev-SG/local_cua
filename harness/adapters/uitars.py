"""UI-TARS-2B-SFT adapter (Transformers on MPS).

Native output: a "Thought: ... Action: ..." block. The Action line is a
pyautogui-style call, e.g.
  Action: click(start_box='<|box_start|>(x,y)<|box_end|>')
with normalised 0-1000 coordinates in the box.
Upstream ships a parser that converts these to PyAutoGUI calls.
"""
from __future__ import annotations

import re
import time

from ..types import Adapter, NormalisedAction, Prediction
from . import register

UITARS_PROMPT = """You are a helpful assistant.

Task: {instruction}
"""


@register("uitars")
class UITarsAdapter(Adapter):
    name = "UI-TARS-2B-SFT"
    family = "qwen2-vl"
    role = "agent"
    artifact = "ByteDance-Seed/UI-TARS-2B-SFT"
    runtime = "transformers-mps"
    quant = "bf16"

    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.artifact)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.artifact, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.model.to("mps")
        self.model.eval()

    def predict(self, image_path, instruction, history=None):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": UITARS_PROMPT.format(instruction=instruction)},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: (v.to("mps") if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        infer_ms = (time.perf_counter() - t0) * 1000
        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(gen, skip_special_tokens=False)[0]

        t1 = time.perf_counter()
        action, err = parse_uitars(raw, self.viewport)
        parse_ms = (time.perf_counter() - t1) * 1000
        return Prediction(
            model=self.name, native_text=raw, action=action,
            parse_ok=err is None, parse_ms=parse_ms, inference_ms=infer_ms,
            point=(action.x, action.y) if action.x is not None else None,
            parse_error=err,
        )


ACTION_RE = re.compile(r"Action:\s*(.*)", re.S)


def parse_uitars(text: str, viewport: dict):
    """Parse UI-TARS Action: line. Normalised coords -> viewport px."""
    m = ACTION_RE.search(text)
    if not m:
        return (NormalisedAction(kind="none", raw={"native": text}),
                "no Action: line")
    body = m.group(1).strip()
    fn = re.match(r"([a-zA-Z_]+)\s*\(", body)
    kind_raw = fn.group(1).lower() if fn else ""

    box = re.search(r"start_box\s*=\s*'?.*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", body)
    x = y = None
    if box:
        x = float(box.group(1)) / 1000.0 * viewport["width"]
        y = float(box.group(2)) / 1000.0 * viewport["height"]

    kind_map = {
        "click": "click", "left_single": "click", "left_double": "double_click",
        "right_single": "right_click", "hover": "move", "drag": "drag",
        "type": "type", "hotkey": "key", "press": "key", "scroll": "scroll",
        "finished": "done", "call_user": "answer", "wait": "wait",
    }
    kind = kind_map.get(kind_raw, "none")
    na = NormalisedAction(kind=kind, x=x, y=y, raw={"action_line": body})
    if kind_raw in ("type",):
        c = re.search(r"content\s*=\s*'([^']*)'", body)
        na.text = c.group(1) if c else ""
    if kind_raw in ("hotkey", "press"):
        c = re.search(r"key\s*=\s*'([^']*)'", body) or re.search(r"key\s*=\s*'([^']*)'", body)
        na.keys = c.group(1) if c else ""
    if kind_raw == "drag":
        e = re.search(r"end_box\s*=\s*'?.*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", body)
        if e:
            na.x2 = float(e.group(1)) / 1000.0 * viewport["width"]
            na.y2 = float(e.group(2)) / 1000.0 * viewport["height"]
    if kind_raw == "scroll":
        c = re.search(r"direction\s*=\s*'(\w+)'", body)
        na.dy = 400 if (c and c.group(1) == "down") else -400
    err = None if kind != "none" else f"unparsed action: {body[:60]!r}"
    return na, err
