"""UI-TARS-2B-SFT adapter (Transformers on MPS).

Native output: "Thought: ... Action: ..." where Action is a pyautogui-style
call using the model's native <point>x y</point> coordinate tags (0-1000).
Upstream parser: codes/ui_tars/prompt.py + the action parser in the UI-TARS repo.
"""
from __future__ import annotations

import re
import time

from ..types import Adapter, NormalisedAction, Prediction
from . import register

# Verbatim COMPUTER_USE_DOUBAO template from bytedance/UI-TARS codes/ui_tars/prompt.py
UITARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
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
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.artifact)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.artifact, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.model.to("mps")
        self.model.eval()

    def predict(self, image_path, instruction, history=None):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        body = UITARS_PROMPT.format(language="English", instruction=instruction)
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
    """Parse UI-TARS Action: line; <point>x y</point> is 0-1000 space."""
    m = ACTION_RE.search(text)
    if not m:
        # tolerate a bare grounding box
        m2 = re.search(r"<\|box_start\|>\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", text)
        if m2:
            x = float(m2.group(1)) / 1000.0 * viewport["width"]
            y = float(m2.group(2)) / 1000.0 * viewport["height"]
            return NormalisedAction(kind="click", x=x, y=y, raw={"bare_box": True}), None
        return (NormalisedAction(kind="none", raw={"native": text}),
                "no Action: line")
    body = m.group(1).strip()
    fn = re.match(r"([a-zA-Z_]+)\s*\(", body)
    kind_raw = fn.group(1).lower() if fn else ""

    def _pt(pat):
        mm = re.search(pat, body)
        if not mm:
            return None, None
        return (float(mm.group(1)) / 1000.0 * viewport["width"],
                float(mm.group(2)) / 1000.0 * viewport["height"])

    # native <point>x y</point> tag, and legacy start_box/box_start forms
    x, y = _pt(r"point\s*=\s*'?<point>\s*([\d.]+)\s+([\d.]+)")
    if x is None:
        x, y = _pt(r"start_box\s*=\s*'?.*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
    if x is None:
        x, y = _pt(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")

    kind_map = {
        "click": "click", "left_single": "click", "left_double": "double_click",
        "right_single": "right_click", "hover": "move", "drag": "drag",
        "type": "type", "hotkey": "key", "press": "key", "scroll": "scroll",
        "finished": "done", "call_user": "answer", "wait": "wait",
    }
    kind = kind_map.get(kind_raw, "none")
    na = NormalisedAction(kind=kind, x=x, y=y, raw={"action_line": body})
    if kind_raw == "type":
        c = re.search(r"content\s*=\s*'([^']*)'", body, re.S)
        na.text = c.group(1) if c else ""
    if kind_raw in ("hotkey", "press"):
        c = re.search(r"key\s*=\s*'([^']*)'", body)
        na.keys = c.group(1) if c else ""
    if kind_raw == "drag":
        e = re.search(r"end_point\s*=\s*'?.*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", body) \
            or re.search(r"<point>\s*([\d.]+)\s+([\d.]+)\s*</point>", body)
        if e:
            na.x2 = float(e.group(1)) / 1000.0 * viewport["width"]
            na.y2 = float(e.group(2)) / 1000.0 * viewport["height"]
    if kind_raw == "scroll":
        c = re.search(r"direction\s*=\s*'(\w+)'", body)
        na.dy = 400 if (c and c.group(1) == "down") else -400
    err = None if kind != "none" else f"unparsed action: {body[:60]!r}"
    return na, err
