"""Metric primitives: grounding scoring + RSS sampling."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass
class GroundingHit:
    hit: bool
    inside: bool
    dist_to_center_px: float
    dist_to_bbox_px: float
    iou_ish: float


def point_in_bbox(x: float, y: float, bbox: dict) -> bool:
    return (bbox["x"] <= x <= bbox["x"] + bbox["w"] and
            bbox["y"] <= y <= bbox["y"] + bbox["h"])


def bbox_distance(x: float, y: float, bbox: dict) -> float:
    """Euclidean distance from the point to the nearest bbox edge (0 if inside)."""
    dx = max(bbox["x"] - x, 0, x - (bbox["x"] + bbox["w"]))
    dy = max(bbox["y"] - y, 0, y - (bbox["y"] + bbox["h"]))
    return float((dx ** 2 + dy ** 2) ** 0.5)


def score_point(x: float, y: float, bbox: dict) -> GroundingHit:
    cx, cy = bbox["cx"], bbox["cy"]
    dist_c = float(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5)
    diag = max((bbox["w"] ** 2 + bbox["h"] ** 2) ** 0.5, 1.0)
    inside = point_in_bbox(x, y, bbox)
    return GroundingHit(
        hit=inside,
        inside=inside,
        dist_to_center_px=dist_c,
        dist_to_bbox_px=bbox_distance(x, y, bbox),
        iou_ish=max(0.0, 1.0 - dist_c / diag),
    )


def rss_mb(pid: int) -> float:
    """Resident set size of a process in MB (macOS ps)."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True
        ).strip()
        return round(int(out) / 1024.0, 2) if out else 0.0
    except Exception:
        return 0.0


def peak_rss_self_mb() -> float:
    """Peak RSS of the current process (ru_maxrss is bytes on macOS)."""
    import os
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)
