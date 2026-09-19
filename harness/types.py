"""Native-output-preserving action model.

Each adapter returns a `Prediction` carrying BOTH the raw native model text
(never rewritten) and a normalised action for the executor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalisedAction:
    kind: str                      # click|double_click|right_click|move|drag|type|key|scroll|select|navigate|back|wait|answer|done|none
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    text: str | None = None
    keys: str | None = None
    dx: float | None = None
    dy: float | None = None
    url: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class Prediction:
    model: str
    native_text: str               # exact model output
    action: NormalisedAction
    parse_ok: bool
    parse_ms: float
    inference_ms: float
    parse_error: str | None = None
    point: tuple[float, float] | None = None   # for grounding scoring (screenshot px)
    extra: dict = field(default_factory=dict)


class Adapter:
    """Base adapter. Subclasses implement load()/predict()."""

    name: str = "base"
    family: str = "unknown"
    role: str = "unknown"          # grounder|actor|agent
    artifact: str = ""
    runtime: str = ""
    quant: str = ""

    def __init__(self, viewport: dict):
        self.viewport = viewport
        self.load_sec: float | None = None

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, image_path: str, instruction: str,
                history: list | None = None) -> Prediction:
        raise NotImplementedError

    def close(self) -> None:
        pass


def template_prompt(processor, artifact: str, prompt: str, num_images: int = 1) -> str:
    """mlx-vlm's generate() does NOT apply the chat template; the CLI does it
    first. Every MLX adapter must template its prompt or the image never
    reaches the model (symptom: constant/echoed output)."""
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    cfg = load_config(artifact)
    return apply_chat_template(processor, cfg, prompt, num_images=num_images)
