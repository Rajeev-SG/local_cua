"""Common isolated Playwright Chromium executor for the local_cua screen.

One process, one fixed viewport (1440x900 by default), no user profile,
no cloud. Device scale factor 1 so CSS px == screenshot px, which keeps
grounding scoring exact.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

DEFAULT_VIEWPORT = {"width": 1440, "height": 900}
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class StepResult:
    ok: bool
    executor_ms: float
    detail: str = ""


@dataclass
class PageState:
    url: str
    screenshot_path: str
    viewport: dict
    scroll: dict
    dom: dict = field(default_factory=dict)


class Executor:
    """Wraps a single Chromium page with a neutral action set."""

    def __init__(self, viewport: dict | None = None, headless: bool = True):
        self.viewport = dict(viewport or DEFAULT_VIEWPORT)
        self.headless = headless
        self._pw = None
        self.browser = None
        self.context = None
        self.page = None

    # ---- lifecycle ----
    def start(self) -> None:
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(
            headless=self.headless,
            args=["--disable-lcd-text", "--force-device-scale-factor=1"],
        )
        self.context = self.browser.new_context(
            viewport=self.viewport,
            device_scale_factor=1,
            reduced_motion="reduce",
            locale="en-US",
        )
        self.context.add_init_script(
            "Math.random = () => 0.5;"  # determinism for any page jitter
        )
        self.page = self.context.new_page()
        self.page.set_default_timeout(8000)

    def stop(self) -> None:
        try:
            if self.context:
                self.context.close()
        finally:
            if self.browser:
                self.browser.close()
            if self._pw:
                self._pw.stop()

    # ---- navigation ----
    def goto(self, url: str, wait_until: str = "load",
             timeout_ms: int | None = None) -> None:
        """Navigate and settle.

        Local fixtures fire `load` immediately, so the default is unchanged.
        Live sites (real-work tasks) never reach a quiet `load` inside a sane
        budget — third-party analytics keep the load event pending — so those
        callers pass `wait_until="domcontentloaded"` plus a longer timeout.
        """
        self.page.goto(url, wait_until=wait_until,
                       timeout=timeout_ms or 8000)
        self.settle()

    def settle(self) -> None:
        """Wait for layout to be stable and screenshots deterministic."""
        try:
            self.page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass
        self.page.evaluate(
            "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
        )
        time.sleep(0.15)

    # ---- observation ----
    def screenshot(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=path, animations="disabled")
        return path

    def bbox(self, selector: str) -> dict | None:
        el = self.page.query_selector(selector)
        if not el:
            return None
        box = el.bounding_box()
        if not box:
            return None
        return {
            "x": box["x"],
            "y": box["y"],
            "w": box["width"],
            "h": box["height"],
            "cx": box["x"] + box["width"] / 2,
            "cy": box["y"] + box["height"] / 2,
        }

    def scroll_state(self) -> dict:
        return self.page.evaluate(
            "() => ({x: window.scrollX, y: window.scrollY, "
            "h: document.documentElement.scrollHeight, "
            "vh: window.innerHeight})"
        )

    # ---- neutral action set ----
    def click(self, x: float, y: float) -> StepResult:
        t0 = time.perf_counter()
        self.page.mouse.click(x, y)
        return StepResult(True, (time.perf_counter() - t0) * 1000, f"click({x},{y})")

    def double_click(self, x: float, y: float) -> StepResult:
        t0 = time.perf_counter()
        self.page.mouse.dblclick(x, y)
        return StepResult(True, (time.perf_counter() - t0) * 1000, f"dblclick({x},{y})")

    def right_click(self, x: float, y: float) -> StepResult:
        t0 = time.perf_counter()
        self.page.mouse.click(x, y, button="right")
        return StepResult(True, (time.perf_counter() - t0) * 1000, f"rclick({x},{y})")

    def move(self, x: float, y: float) -> StepResult:
        t0 = time.perf_counter()
        self.page.mouse.move(x, y)
        return StepResult(True, (time.perf_counter() - t0) * 1000, f"move({x},{y})")

    def drag(self, x1, y1, x2, y2) -> StepResult:
        t0 = time.perf_counter()
        self.page.mouse.move(x1, y1)
        self.page.mouse.down()
        self.page.mouse.move(x2, y2, steps=8)
        self.page.mouse.up()
        return StepResult(True, (time.perf_counter() - t0) * 1000, "drag")

    def type_text(self, text: str) -> StepResult:
        t0 = time.perf_counter()
        self.page.keyboard.type(text, delay=8)
        return StepResult(True, (time.perf_counter() - t0) * 1000, f"type({text!r})")

    def key(self, keys: str) -> StepResult:
        t0 = time.perf_counter()
        self.page.keyboard.press(keys)
        return StepResult(True, (time.perf_counter() - t0) * 1000, f"key({keys})")

    def scroll(self, dx: float, dy: float) -> StepResult:
        t0 = time.perf_counter()
        self.page.mouse.wheel(dx, dy)
        return StepResult(True, (time.perf_counter() - t0) * 1000, f"scroll({dx},{dy})")

    def select(self, selector: str, value: str) -> StepResult:
        t0 = time.perf_counter()
        self.page.select_option(selector, value)
        return StepResult(True, (time.perf_counter() - t0) * 1000, "select")

    def navigate(self, url: str, wait_until: str = "load",
                 timeout_ms: int | None = None) -> StepResult:
        t0 = time.perf_counter()
        self.goto(url, wait_until=wait_until, timeout_ms=timeout_ms)
        return StepResult(True, (time.perf_counter() - t0) * 1000, f"goto({url})")

    def eval_js(self, expression: str):
        """Run a verification expression in the page (harness-side truth only).

        Used by the real-work runner to recompute the corpus task's objective
        end state; models never get this — Fara's action space has no eval.
        """
        return self.page.evaluate(expression)

    def back(self) -> StepResult:
        t0 = time.perf_counter()
        self.page.go_back(wait_until="load")
        self.settle()
        return StepResult(True, (time.perf_counter() - t0) * 1000, "back")


def fixture_url(name: str) -> str:
    return (FIXTURES / name).resolve().as_uri()
