"""TongUI-3B adapter (Transformers on MPS).

Native output: JSON action(s); web actions include CLICK, INPUT, SELECT,
HOVER, ANSWER, ENTER, SCROLL, SELECT_TEXT, COPY. Coordinates are absolute
pixels of the (possibly resized) screenshot; TongUI emits them in the
resized-image space, so we rescale to the viewport.
May emit multiple actions per call — preserve them.
"""
from __future__ import annotations

import json
import re
import time

from ..types import Adapter, NormalisedAction, Prediction
from . import register


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
                {"type": "text", "text": instruction},
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
        actions, err = parse_tongui(raw, self.viewport, img.size)
        parse_ms = (time.perf_counter() - t1) * 1000
        first = actions[0] if actions else NormalisedAction(kind="none")
        return Prediction(
            model=self.name, native_text=raw, action=first,
            parse_ok=err is None, parse_ms=parse_ms, inference_ms=infer_ms,
            point=(first.x, first.y) if first.x is not None else None,
            parse_error=err, extra={"n_actions": len(actions),
                                    "all_actions": [a.__dict__ for a in actions]},
        )


def parse_tongui(text: str, viewport: dict, img_size: tuple):
    """Parse one-or-more TongUI JSON actions. Returns (list, error)."""
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            continue
    if not objs:
        return ([], "no JSON action found")
    # TongUI coordinates are absolute pixels in the screenshot it saw.
    sx = viewport["width"] / img_size[0]
    sy = viewport["height"] / img_size[1]
    kind_map = {
        "CLICK": "click", "INPUT": "type", "SELECT": "click", "HOVER": "move",
        "ANSWER": "answer", "ENTER": "key", "SCROLL": "scroll",
        "SELECT_TEXT": "click", "COPY": "key",
    }
    out = []
    for d in objs:
        act = str(d.get("action", d.get("Action", ""))).upper()
        coord = d.get("coordinate", d.get("coordinates", d.get("position")))
        x = y = None
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            x = float(coord[0]) * sx
            y = float(coord[1]) * sy
        elif isinstance(coord, dict):
            x = float(coord.get("x", 0)) * sx
            y = float(coord.get("y", 0)) * sy
        na = NormalisedAction(kind=kind_map.get(act, "none"), x=x, y=y, raw=d)
        val = d.get("value", d.get("text", ""))
        if act == "INPUT":
            na.text = str(val)
        if act == "ENTER":
            na.keys = "Enter"
        if act == "COPY":
            na.keys = "Meta+C"
        if act == "SCROLL":
            na.dy = float(d.get("value", 400) or 400)
        if act == "ANSWER":
            na.text = str(val)
        out.append(na)
    return (out, None)
