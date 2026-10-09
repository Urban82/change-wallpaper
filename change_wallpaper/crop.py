# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Danilo Treffiletti

from dataclasses import dataclass, field

from change_wallpaper.detection import Box
from change_wallpaper.log import get_logger

log = get_logger(__name__)

# Images whose aspect ratio is within this tolerance of 1:1 are "square"
SQUARE_TOLERANCE = 0.05
# Subjects with an area below this fraction of the biggest one are background
BACKGROUND_FACE_RATIO = 0.25
# Top part of a person box used when the whole box does not fit the crop
PERSON_HEAD_FRACTION = 0.25


@dataclass
class Subjects:
    """Detected subjects of an image: one kind only, split into foreground (used) and background (ignored).

    kind is "face", "person", "animal", "vehicle", or None when nothing was found.
    """

    kind: str | None = None
    main: list[Box] = field(default_factory=list)
    background: list[Box] = field(default_factory=list)


def is_square(width: int, height: int) -> bool:
    return abs(width / height - 1.0) <= SQUARE_TOLERANCE


def foreground(boxes: list[Box], ratio: float = BACKGROUND_FACE_RATIO,
               kind: str = "object") -> list[Box]:
    """Drop background boxes: those whose area is below ratio * the biggest area."""
    if len(boxes) < 2 or ratio <= 0:
        return boxes
    biggest = max(b.area for b in boxes)
    kept = [b for b in boxes if b.area >= biggest * ratio]
    if len(kept) < len(boxes):
        log.debug("Ignoring %d background %s(s) smaller than %.0f%% of the biggest",
                  len(boxes) - len(kept), kind, ratio * 100)
    return kept


def layout_orientation(boxes: list[Box]) -> str | None:
    """Preferred orientation given the boxes layout: 'horizontal', 'vertical' or None."""
    if len(boxes) < 2:
        return None
    x0 = min(b.x for b in boxes)
    x1 = max(b.x + b.w for b in boxes)
    y0 = min(b.y for b in boxes)
    y1 = max(b.y + b.h for b in boxes)
    span_w, span_h = x1 - x0, y1 - y0
    log.debug("Subjects span %.0fx%.0f", span_w, span_h)
    if span_w > span_h * 1.2:
        return "horizontal"
    if span_h > span_w * 1.2:
        return "vertical"
    return None


def _best_offset(size: int, length: int, spans: list[tuple[float, float, float]]) -> int:
    """Best window [p, p+length] within [0, size] containing the most spans.

    Each span is (start, end, weight). The window is centered on the
    weighted center of the contained spans, as far as it can be while
    still containing all of them.
    """
    max_p = size - length
    if max_p <= 0:
        return 0
    if not spans:
        return max_p // 2

    def clamp(p: float, lo: float = 0.0, hi: float = float(max_p)) -> float:
        return min(max(p, lo), hi)

    candidates = {clamp(a) for a, _, _ in spans} | {clamp(b - length) for _, b, _ in spans}
    best_key, best_p = None, max_p / 2
    for p in candidates:
        inside = [s for s in spans if s[0] >= p - 0.5 and s[1] <= p + length + 0.5]
        if not inside:
            continue
        total = sum(w for _, _, w in inside)
        key = (len(inside), total)
        if best_key is not None and key <= best_key:
            continue
        center = sum((a + b) / 2 * w for a, b, w in inside) / total
        # window positions that keep every contained span inside
        lo = max(0.0, max(b for _, b, _ in inside) - length)
        hi = min(float(max_p), min(a for a, _, _ in inside))
        best_key, best_p = key, clamp(center - length / 2, lo, max(lo, hi))
    return round(best_p)


def _fit_span(a: float, b: float, length: int) -> tuple[float, float]:
    """Span itself if it fits in the window, otherwise its center point."""
    if b - a <= length:
        return a, b
    mid = (a + b) / 2
    return mid, mid


def _top_span(start: float, size: float, length: int) -> tuple[float, float]:
    """Whole box if it fits; otherwise its top part (a person's head and shoulders)."""
    if size <= length:
        return start, start + size
    return _fit_span(start, start + size * PERSON_HEAD_FRACTION, length)


def crop_box(width: int, height: int, aspect: float, boxes: list[Box],
             top_fallback: bool = False) -> tuple[int, int, int, int]:
    """Largest box with the given aspect that keeps the most subjects.

    Larger (closer) subjects weigh more when centering the box. Boxes are kept
    whole when possible; otherwise, with top_fallback (people), the top part
    of the box is used on the y axis, else its center.

    Only one axis is ever cropped: either left/right or top/bottom.
    """
    if width / height > aspect:
        cw = max(1, min(width, round(height * aspect)))
        spans = [(*_fit_span(b.x, b.x + b.w, cw), b.area) for b in boxes]
        x = _best_offset(width, cw, spans)
        return x, 0, x + cw, height
    ch = max(1, min(height, round(width / aspect)))
    spans = [(*(_top_span(b.y, b.h, ch) if top_fallback else _fit_span(b.y, b.y + b.h, ch)), b.area)
             for b in boxes]
    y = _best_offset(height, ch, spans)
    return 0, y, width, y + ch
