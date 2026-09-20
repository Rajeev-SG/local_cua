"""Tests for the Stage-3 real-work runner (frontier findings F2/F3)."""
import json
import sys
from pathlib import Path


def _load():
    # Import as part of the `harness` package so the module's relative imports
    # (`from .browser import ...`) resolve.
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import harness.run_stage3 as m
    return m


def test_vendored_corpus_present_and_matches_manifest():
    """F2: Stage 3 must be reproducible offline from the vendored corpus."""
    root = Path(__file__).resolve().parent.parent
    vend = root / "corpus" / "microbench"
    assert (vend / "SOURCE.json").exists()
    manifest = json.loads((vend / "SOURCE.json").read_text())
    import hashlib
    for tid, meta in manifest["tasks"].items():
        f = vend / "tasks" / f"{tid}.json"
        assert f.exists(), f"vendored task missing: {tid}"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == meta["sha256"], tid
    assert (vend / "pass_rule.py").exists()


def test_terminate_relay_recognised():
    """F3: a native `terminate` action must open the answer relay.

    The model's own action space is terminate/read_page_answer_question; the
    runner must not depend on `done`/`answer` alone, or every task would look
    unpassable.
    """
    m = _load()
    from harness.adapters.fara import parse_fara
    viewport = {"width": 1440, "height": 900}
    text = ('<tool_call>{"name": "computer_use", "arguments": '
            '{"action": "terminate", "answer": "{\\"gtm\\":[\\"GA4\\"]}"}}</tool_call>')
    act, err = parse_fara(text, viewport)
    # normalised kind
    assert act.kind == "done", act.kind
    # raw native action still visible for the F3 belt-and-braces check
    assert act.raw["arguments"]["action"] == "terminate"
    # the runner's break condition must accept it
    raw_action = (act.raw or {}).get("arguments", {}).get("action")
    assert act.kind in ("done", "answer") or raw_action in (
        "terminate", "read_page_answer_question", "ask_user_question")


def test_extract_json_from_answer():
    m = _load()
    assert m._extract_json('{"a": 1}') == {"a": 1}
    assert m._extract_json('prose {"a": 1} more') == {"a": 1}
    assert m._extract_json("no json") is None
