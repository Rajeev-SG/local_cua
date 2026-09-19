"""UGround-V1-2B grounder adapter (mlx-vlm).

Native output: a single "(x, y)" string in a [0,1000) coordinate space.
It is a grounder only: the adapter never plans.
"""
from __future__ import annotations

import re
import time

from ..types import Adapter, NormalisedAction, Prediction, template_prompt
from . import register

UGROUND_PROMPT = """
  Your task is to help the user identify the precise coordinates (x, y) of a specific area on the screen based on a description.

  - Your response should aim to point should be the center of the described area or object.
  - If the description is unclear or ambiguous, infer the most relevant area or object based on its likely context or purpose.
  - Your answer should be a single string (x, y) corresponding to the point of the interest.

  Description: {description}

  Answer:"""


@register("uground")
class UGroundAdapter(Adapter):
    name = "UGround-V1-2B"
    family = "qwen2-vl"
    role = "grounder"
    artifact = "mlx-community/UGround-V1-2B-bf16"
    runtime = "mlx-vlm"
    quant = "bf16"

    def load(self):
        from mlx_vlm import load
        self.model, self.processor = load(self.artifact)

    def predict(self, image_path, instruction, history=None):
        from mlx_vlm import generate
        prompt = template_prompt(self.processor, self.artifact,
                                 UGROUND_PROMPT.format(description=instruction))
        t0 = time.perf_counter()
        res = generate(self.model, self.processor, prompt, image=image_path,
                       max_tokens=24, temperature=0.0, verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000
        text = res.text if hasattr(res, "text") else str(res)

        t1 = time.perf_counter()
        pt = parse_point_1000(text)
        x = y = None
        if pt:
            x = pt[0] / 1000.0 * self.viewport["width"]
            y = pt[1] / 1000.0 * self.viewport["height"]
        parse_ms = (time.perf_counter() - t1) * 1000

        return Prediction(
            model=self.name,
            native_text=text,
            action=NormalisedAction(kind="click" if pt else "none", x=x, y=y,
                                    raw={"space": "0-1000"}),
            parse_ok=pt is not None,
            parse_ms=parse_ms,
            inference_ms=infer_ms,
            point=(x, y) if pt else None,
            parse_error=None if pt else "no (x, y) found",
        )


def parse_point_1000(text: str):
    m = re.search(r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)", text)
    if not m:
        # tolerate bare "123, 456"
        m = re.search(r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))
