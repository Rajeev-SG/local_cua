"""Fara1.5-4B adapter (mlx-vlm) — native browser CUA.

Native output: chain-of-thought text followed by
  <tool_call>{"name": "computer_use", "arguments": {"action": ...}}</tool_call>
Coordinates are emitted in a fixed 1000x1000 space and scaled to the viewport.
Uses Microsoft's verbatim system prompt + computer_use tool schema.
"""
from __future__ import annotations

import json
import re
import time

from ..types import Adapter, NormalisedAction, Prediction, template_prompt
from . import register

IDENTITY = """You are Fara, a computer use agent (CUA) specialized for web browsers. \
You are developed by Microsoft AI Frontiers. You assist users with \
completing and automating tasks that require the use of a web browser.

The model was trained in the timeframe of January - April 2026. You can \
effectively perform tasks even beyond this range by accessing the web \
browser and using the latest information on the live web. But your \
knowledge cutoff is limited to early 2026, so you may not be aware of \
events or developments that occurred after that time, without explicitly \
browsing and searching for latest information on the web.

This edition of the model was trained using SFT on top of Qwen3.5-4B, \
using a synthetic data mixture generated and developed by Microsoft AI Frontiers."""

CRITICAL_POINTS = """\
A critical point is a situation where we must pause and request information or confirmation from the user before \
proceeding. There are three types:

Case 1: Missing User Information — The task requires personal information that the user has not provided (e.g., email, \
phone number, address, payment details). Never fabricate or assume personal information. Fill in only what the user has \
explicitly provided, then pause and ask for any missing required fields.

Case 2: Underspecified Task — The task description is ambiguous or missing details needed to make a decision at the \
current step. Pause and ask for clarification.

Case 3: Irreversible Action — We are about to perform an action that cannot be undone (e.g., submitting a form, \
completing a purchase, sending a message, deleting data). If the user explicitly authorized the action, proceed. \
Otherwise, stop and ask for confirmation.

Only stop at a critical point if (1) required information is missing, (2) the task is ambiguous, OR (3) an irreversible \
action lacks explicit user authorization."""

FN_CALL_FORMAT = (
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    "__TOOL_DESCS__\n"
    "</tools>\n"
    "\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

_ACTIONS = ["key", "type", "mouse_move", "left_click", "left_click_drag",
            "right_click", "double_click", "triple_click", "scroll", "hscroll",
            "visit_url", "history_back", "web_search", "read_page_answer_question",
            "pause_and_memorize_fact", "ask_user_question", "wait", "terminate"]


def build_system_prompt(width: int, height: int) -> str:
    desc = f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {width}x{height}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.
""".strip()
    tool_json = {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": desc,
            "parameters": {
                "properties": {
                    "action": {"description": "The action to perform.", "enum": _ACTIONS, "type": "string"},
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "(x, y) pixel coordinate.", "type": "array"},
                    "pixels": {"description": "Scroll amount.", "type": "number"},
                    "url": {"description": "URL for visit_url.", "type": "string"},
                    "query": {"description": "Search query.", "type": "string"},
                    "fact": {"description": "Fact to memorize.", "type": "string"},
                    "question": {"description": "Question to ask.", "type": "string"},
                    "time": {"description": "Seconds to wait.", "type": "number"},
                    "answer": {"description": "Final answer.", "type": "string"},
                },
                "required": ["action"],
                "type": "object",
            },
        },
    }
    return (IDENTITY + "\n\n" + CRITICAL_POINTS + "\n\n" +
            FN_CALL_FORMAT.replace("__TOOL_DESCS__", json.dumps(tool_json)))


@register("fara")
class FaraAdapter(Adapter):
    name = "Fara1.5-4B"
    family = "qwen3.5-vl"
    role = "agent"
    artifact = "runanywhere/Fara1.5-4B-mlx-4bit"
    runtime = "mlx-vlm"
    quant = "4bit (vision tower bf16)"

    def load(self):
        from mlx_vlm import load
        self.model, self.processor = load(self.artifact)
        self.system_prompt = build_system_prompt(
            self.viewport["width"], self.viewport["height"])

    def predict(self, image_path, instruction, history=None):
        from mlx_vlm import generate
        hist = ""
        if history:
            hist = "".join(f"\n{role}: {txt}" for role, txt in history[-4:])
        prompt = (f"{self.system_prompt}\n\nTask: {instruction}{hist}\n")
        prompt = template_prompt(self.processor, self.artifact, prompt)
        t0 = time.perf_counter()
        res = generate(self.model, self.processor, prompt, image=image_path,
                       max_tokens=512, temperature=0.0)
        infer_ms = (time.perf_counter() - t0) * 1000
        text = res.text if hasattr(res, "text") else str(res)

        t1 = time.perf_counter()
        action, err = parse_fara(text, self.viewport)
        parse_ms = (time.perf_counter() - t1) * 1000
        return Prediction(
            model=self.name, native_text=text, action=action,
            parse_ok=err is None, parse_ms=parse_ms, inference_ms=infer_ms,
            point=(action.x, action.y) if action.x is not None else None,
            parse_error=err,
        )


def parse_fara(text: str, viewport: dict):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        return (NormalisedAction(kind="none", raw={"native": text}),
                "no <tool_call> block")
    try:
        d = json.loads(m.group(1))
    except Exception as e:
        return (NormalisedAction(kind="none", raw={"native": text}),
                f"json decode: {e}")
    args = d.get("arguments", {}) or {}
    act = args.get("action", "")
    coord = args.get("coordinate") or []
    x = y = None
    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
        # Fara always predicts in a FIXED 1000x1000 space (Microsoft model card:
        # "coordinates are returned in a fixed 1000x1000 space; scale to your
        # viewport"). This is viewport-independent: verified empirically that the
        # model returns the same coordinate when told 1440x900 or 1000x1000
        # (tests/test_parsers.py::test_fara_space_is_viewport_independent).
        x = float(coord[0]) / 1000.0 * viewport["width"]
        y = float(coord[1]) / 1000.0 * viewport["height"]
    kind_map = {
        "left_click": "click", "double_click": "double_click",
        "right_click": "right_click", "triple_click": "click",
        "mouse_move": "move", "left_click_drag": "drag",
        "type": "type", "key": "key", "scroll": "scroll", "hscroll": "scroll",
        "visit_url": "navigate", "history_back": "back", "web_search": "navigate",
        "wait": "wait", "terminate": "done", "ask_user_question": "answer",
        "read_page_answer_question": "answer",
        "pause_and_memorize_fact": "none",
    }
    na = NormalisedAction(kind=kind_map.get(act, "none"), x=x, y=y, raw=d)
    if act == "type":
        na.text = str(args.get("text", ""))
    if act == "key":
        keys = args.get("keys") or []
        na.keys = "+".join(keys) if isinstance(keys, list) else str(keys)
    if act in ("scroll", "hscroll"):
        # Fara schema: "Positive values scroll up, negative values scroll down."
        # Playwright mouse.wheel: positive dy scrolls DOWN -> invert.
        px = float(args.get("pixels", 300) or 300)
        na.dy = -px if act == "scroll" else 0.0
        na.dx = px if act == "hscroll" else 0.0
    if act == "visit_url":
        na.url = str(args.get("url", ""))
    if act in ("terminate", "ask_user_question", "read_page_answer_question"):
        na.text = str(args.get("answer", args.get("question", "")))
    err = None if act in _ACTIONS else f"unknown action {act!r}"
    return na, err
