"""Table-driven tests for the pure parsing/normalisation/scoring functions.

These parsers decide every reported score, so their coordinate-space
heuristics and edge cases are pinned here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.adapters.fara import parse_fara
from harness.adapters.showui import parse_showui
from harness.adapters.tongui import parse_tongui
from harness.adapters.uitars import parse_uitars
from harness.adapters.uground import parse_point_1000
from harness.metrics import point_in_bbox, score_point, bbox_distance

V = {"width": 1440, "height": 900}


# ---------------- Fara: 1000x1000 space -> viewport ----------------
def test_fara_click_scales_1000_space():
    txt = ('<tool_call>\n{"name": "computer_use", '
           '"arguments": {"action": "left_click", "coordinate": [500, 500]}}\n</tool_call>')
    a, err = parse_fara(txt, V)
    assert err is None
    assert a.kind == "click"
    assert a.x == 720 and a.y == 450          # 500/1000 * 1440, 500/1000 * 900


def test_fara_type_and_terminate():
    txt = ('<tool_call>{"name": "computer_use", "arguments": '
           '{"action": "type", "text": "hello"}}</tool_call>')
    a, err = parse_fara(txt, V)
    assert err is None and a.kind == "type" and a.text == "hello"

    txt2 = ('<tool_call>{"name": "computer_use", "arguments": '
            '{"action": "terminate", "answer": "done"}}</tool_call>')
    a2, err2 = parse_fara(txt2, V)
    assert err2 is None and a2.kind == "done" and a2.text == "done"


def test_fara_rejects_missing_and_malformed():
    a, err = parse_fara("no tool call here", V)
    assert err is not None and a.kind == "none"
    a2, err2 = parse_fara("<tool_call>{not json}</tool_call>", V)
    assert err2 is not None and a2.kind == "none"


def test_fara_unknown_action_is_error():
    txt = '<tool_call>{"name": "computer_use", "arguments": {"action": "frobnicate"}}</tool_call>'
    a, err = parse_fara(txt, V)
    assert err is not None and a.kind == "none"


# ---------------- ShowUI: relative 0-1 and dict/list forms ----------------
def test_showui_nav_dict_relative():
    a, err = parse_showui("{'action': 'CLICK', 'value': None, 'position': [0.5, 0.5]}", V, "direct")
    assert err is None and a.kind == "click"
    assert a.x == 720 and a.y == 450


def test_showui_drag_nested_position():
    txt = ("{'action': 'SELECT_TEXT', 'value': None, "
           "'position': [[0.1, 0.2], [0.3, 0.4]]}")
    a, err = parse_showui(txt, V, "direct")
    assert a.kind == "click"
    assert a.x == 144 and a.y == 180     # start of the selection


def test_showui_input_sets_text():
    a, err = parse_showui("{'action': 'INPUT', 'value': 'hi', 'position': [0.5, 0.5]}", V, "direct")
    assert a.kind == "type" and a.text == "hi"


def test_showui_actor_list_form():
    a, err = parse_showui("[0.25, 0.5]", V, "actor")
    assert err is None and a.kind == "click"
    assert a.x == 360 and a.y == 450


def test_showui_list_form_flagged_in_direct_mode():
    a, err = parse_showui("[0.25, 0.5]", V, "direct")
    assert err is not None and a.kind == "click"   # valid point but wrong form for direct


def test_showui_malformed():
    a, err = parse_showui("totally not an action", V, "direct")
    assert err is not None and a.kind == "none"


# ---------------- TongUI: relative 0-1, Thought/Action JSON ----------------
def test_tongui_single_action():
    txt = 'Thought: click it.\nAction: {"action": "CLICK", "value": null, "position": [0.08, 0.18]}'
    acts, err = parse_tongui(txt, V)
    assert err is None and len(acts) == 1
    assert acts[0].kind == "click"
    assert round(acts[0].x) == 115 and round(acts[0].y) == 162


def test_tongui_multi_action_array():
    txt = ('Action: [{"action": "CLICK", "value": null, "position": [0.5, 0.5]}, '
           '{"action": "INPUT", "value": "abc", "position": [0.5, 0.5]}]')
    acts, err = parse_tongui(txt, V)
    assert err is None and len(acts) == 2
    assert acts[0].kind == "click" and acts[1].kind == "type" and acts[1].text == "abc"


def test_tongui_scroll_direction():
    txt = 'Action: {"action": "SCROLL", "value": "down", "position": [0.5, 0.5]}'
    acts, _ = parse_tongui(txt, V)
    assert acts[0].dy == 500
    txt2 = 'Action: {"action": "SCROLL", "value": "up", "position": [0.5, 0.5]}'
    acts2, _ = parse_tongui(txt2, V)
    assert acts2[0].dy == -500


def test_tongui_malformed():
    acts, err = parse_tongui("no json at all", V)
    assert err is not None and acts == []


# ---------------- UI-TARS: <point>x y</point> 0-1000 space ----------------
def test_uitars_point_tag():
    txt = "Thought: go.\nAction: click(point='<point>500 500</point>')"
    a, err = parse_uitars(txt, V)
    assert err is None and a.kind == "click"
    assert a.x == 720 and a.y == 450


def test_uitars_bare_box_tolerated():
    a, err = parse_uitars("<|box_start|>(500,500)<|box_end|>", V)
    assert err is None and a.kind == "click" and a.x == 720


def test_uitars_type_content():
    txt = "Action: type(content='hello world')"
    a, err = parse_uitars(txt, V)
    assert err is None and a.kind == "type" and a.text == "hello world"


def test_uitars_malformed():
    a, err = parse_uitars("Thought: nothing useful", V)
    assert err is not None and a.kind == "none"


# ---------------- UGround: (x, y) 0-1000 ----------------
def test_uground_point():
    assert parse_point_1000(" (166, 176)") == (166.0, 176.0)
    assert parse_point_1000("Answer: 12, 34") == (12.0, 34.0)


def test_uground_no_point():
    assert parse_point_1000("I cannot find it") is None


# ---------------- metrics: scoring self-tests ----------------
BBOX = {"x": 100, "y": 100, "w": 200, "h": 100, "cx": 200, "cy": 150}


def test_point_in_bbox_known_hit_and_miss():
    assert point_in_bbox(200, 150, BBOX) is True
    assert point_in_bbox(100, 100, BBOX) is True      # corner is inside
    assert point_in_bbox(50, 150, BBOX) is False
    assert point_in_bbox(350, 150, BBOX) is False
    assert point_in_bbox(200, 250, BBOX) is False


def test_bbox_distance_zero_inside_and_positive_outside():
    assert bbox_distance(200, 150, BBOX) == 0.0
    assert bbox_distance(50, 150, BBOX) == 50.0
    assert bbox_distance(200, 220, BBOX) == 20.0


def test_score_point_hit_and_center_distance():
    hit = score_point(200, 150, BBOX)
    assert hit.hit is True and hit.dist_to_center_px == 0.0
    # x=350 is outside the 100..300 bbox (inclusive edges)
    miss = score_point(350, 150, BBOX)
    assert miss.hit is False
    assert miss.dist_to_center_px == 150.0
    assert miss.dist_to_bbox_px == 50.0


# ---------------- per-scene scoring self-test (known hit / known miss) ----------------
import json


def _fixtures():
    idx = Path(__file__).resolve().parent.parent / "results/fixtures/stage1/index.json"
    if not idx.exists():
        return {}
    return json.loads(idx.read_text())


def test_stage1_known_hit_and_miss_per_scene():
    """Each scene: the recorded DOM centre must score a hit, and centre + far
    offset must score a miss. Guards against a flipped/serialised bbox.

    The fixture index (bbox metadata only, no screenshots) is committed, so this
    must run in CI - it fails loudly rather than skipping if the index is absent.
    """
    fx = _fixtures()
    assert fx, ("stage1 fixture index missing: expected "
                "results/fixtures/stage1/index.json to be committed")
    assert len(fx) == 8, f"expected 8 scenes, got {len(fx)}"
    for name, meta in fx.items():
        b = meta["bbox"]
        assert point_in_bbox(b["cx"], b["cy"], b), f"{name}: centre should hit"
        assert not point_in_bbox(b["cx"] + b["w"], b["cy"] + b["h"], b), \
            f"{name}: far corner should miss"
        hit = score_point(b["cx"], b["cy"], b)
        assert hit.hit and hit.dist_to_center_px == 0.0


# ---------------- Fara coordinate-space + scroll regressions (review F1/F2) ----------------
def test_fara_space_is_viewport_independent():
    """Fara predicts in a FIXED 1000x1000 space regardless of viewport.
    The same raw coordinate must scale differently per viewport - i.e. the
    adapter must NOT treat the number as already being viewport pixels."""
    txt = ('<tool_call>{"name": "computer_use", "arguments": '
           '{"action": "left_click", "coordinate": [196, 175]}}</tool_call>')
    a_1440, _ = parse_fara(txt, {"width": 1440, "height": 900})
    a_1280, _ = parse_fara(txt, {"width": 1280, "height": 800})
    assert round(a_1440.x, 1) == 282.2 and round(a_1440.y, 1) == 157.5
    assert round(a_1280.x, 1) == 250.9 and round(a_1280.y, 1) == 140.0
    assert a_1440.x != a_1280.x, "scaling must depend on the viewport"


def test_fara_scroll_direction_passthrough():
    """Fara: positive pixels = scroll UP. Playwright: positive dy = scroll DOWN.
    The adapter must invert, and upward scroll must be representable."""
    up = ('<tool_call>{"name": "computer_use", "arguments": '
          '{"action": "scroll", "pixels": 400}}</tool_call>')
    a_up, _ = parse_fara(up, V)
    assert a_up.dy == -400, "scroll up must become negative dy"

    down = ('<tool_call>{"name": "computer_use", "arguments": '
            '{"action": "scroll", "pixels": -400}}</tool_call>')
    a_down, _ = parse_fara(down, V)
    assert a_down.dy == 400

    h = ('<tool_call>{"name": "computer_use", "arguments": '
         '{"action": "hscroll", "pixels": 200}}</tool_call>')
    a_h, _ = parse_fara(h, V)
    assert a_h.dx == 200 and a_h.dy == 0.0
