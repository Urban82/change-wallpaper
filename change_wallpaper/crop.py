from change_wallpaper.faces import Face
from change_wallpaper.log import get_logger

log = get_logger(__name__)

# Images whose aspect ratio is within this tolerance of 1:1 are "square"
SQUARE_TOLERANCE = 0.05
# Faces with an area below this fraction of the biggest face are background
BACKGROUND_FACE_RATIO = 0.25


def is_square(width: int, height: int) -> bool:
    return abs(width / height - 1.0) <= SQUARE_TOLERANCE


def foreground_faces(faces: list[Face], ratio: float = BACKGROUND_FACE_RATIO) -> list[Face]:
    """Drop background faces: those whose area is below ratio * the biggest face area."""
    if len(faces) < 2 or ratio <= 0:
        return faces
    biggest = max(f.w * f.h for f in faces)
    kept = [f for f in faces if f.w * f.h >= biggest * ratio]
    if len(kept) < len(faces):
        log.debug("Ignoring %d background face(s) smaller than %.0f%% of the biggest",
                  len(faces) - len(kept), ratio * 100)
    return kept


def face_orientation(faces: list[Face]) -> str | None:
    """Preferred orientation given face layout: 'horizontal', 'vertical' or None."""
    if len(faces) < 2:
        return None
    x0 = min(f.x for f in faces)
    x1 = max(f.x + f.w for f in faces)
    y0 = min(f.y for f in faces)
    y1 = max(f.y + f.h for f in faces)
    span_w, span_h = x1 - x0, y1 - y0
    log.debug("Face span %.0fx%.0f", span_w, span_h)
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


def crop_box(width: int, height: int, aspect: float, faces: list[Face]) -> tuple[int, int, int, int]:
    """Largest box with the given aspect that keeps the most faces.

    Larger (closer) faces weigh more when centering the box.

    Only one axis is ever cropped: either left/right or top/bottom.
    """
    if width / height > aspect:
        cw = max(1, min(width, round(height * aspect)))
        x = _best_offset(width, cw, [(f.x, f.x + f.w, f.w * f.h) for f in faces])
        return x, 0, x + cw, height
    ch = max(1, min(height, round(width / aspect)))
    y = _best_offset(height, ch, [(f.y, f.y + f.h, f.w * f.h) for f in faces])
    return 0, y, width, y + ch
