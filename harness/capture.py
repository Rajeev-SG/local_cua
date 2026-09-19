"""Capture deterministic Stage-1 grounding screenshots + DOM target bboxes.

Output: results/fixtures/stage1/<scene>/{shot.png, meta.json}
"""
from __future__ import annotations

import json
from pathlib import Path

from .browser import Executor, fixture_url, DEFAULT_VIEWPORT

OUT = Path(__file__).resolve().parent.parent / "results" / "fixtures" / "stage1"

# scene id -> (instruction, target selector)
SCENES = {
    "1": ("Click the 'Save changes' button.", "#t1_primary"),
    "2": ("Click the email address text field.", "#t2_email"),
    "3": ("Click the checkbox labelled 'I accept the terms'.", "#t3_terms"),
    "4": ("Click the 'Active' filter tab.", "#t4_active"),
    "5": ("Open the country dropdown.", "#t5_country"),
    "6": ("Click the small link 'See full terms and conditions'.", "#t6_terms_link"),
    "7": ("Click the 'Confirm at bottom' button.", "#t7_bottom"),
    "8": ("Click the second 'Delete account' button (the right-hand one).", "#t8_delete_b"),
}


def main():
    ex = Executor()
    ex.start()
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    try:
        for sid, (instr, sel) in SCENES.items():
            d = OUT / f"scene{sid}"
            d.mkdir(parents=True, exist_ok=True)
            ex.goto(fixture_url("scenes.html") + f"?scene={sid}")
            if sid == "7":
                # Scene 7 is the "target after scroll" case: scroll the target
                # into view so it is a fair, visible grounding target.
                ex.page.locator(sel).scroll_into_view_if_needed()
            ex.settle()
            shot = ex.screenshot(str(d / "shot.png"))
            bbox = ex.bbox(sel)
            meta = {
                "scene": sid,
                "instruction": instr,
                "selector": sel,
                "bbox": bbox,
                "viewport": ex.viewport,
                "scroll": ex.scroll_state(),
                "screenshot": shot,
            }
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            index[f"scene{sid}"] = meta
            print(f"scene{sid}: bbox={bbox} scrollY={meta['scroll']['y']}")
    finally:
        ex.stop()
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(index)} scenes -> {OUT}")


if __name__ == "__main__":
    main()
