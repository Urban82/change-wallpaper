from change_wallpaper.faces import Face
from change_wallpaper.log import get_logger

log = get_logger(__name__)

# Images whose aspect ratio is within this tolerance of 1:1 are "square"
SQUARE_TOLERANCE = 0.05


def is_square(width: int, height: int) -> bool:
    return abs(width / height - 1.0) <= SQUARE_TOLERANCE


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


def _best_offset(size: int, length: int, spans: list[tuple[float, float]]) -> int:
    """Best window [p, p+length] within [0, size] containing the most spans."""
    max_p = size - length
    if max_p <= 0:
        return 0
    if not spans:
        return max_p // 2

    def clamp(p: float) -> float:
        return min(max(p, 0.0), float(max_p))

    candidates = {clamp(a) for a, _ in spans} | {clamp(b - length) for _, b in spans}
    best_key, best_p = None, max_p / 2
    for p in candidates:
        inside = [(a, b) for a, b in spans if a >= p - 0.5 and b <= p + length + 0.5]
        if not inside:
            continue
        lo = min(a for a, _ in inside)
        hi = max(b for _, b in inside)
        # center the contained faces in the window (still contains them all)
        centered = clamp((lo + hi) / 2 - length / 2)
        key = (len(inside), sum(b - a for a, b in inside))
        if best_key is None or key > best_key:
            best_key, best_p = key, centered
    return round(best_p)


def crop_box(width: int, height: int, aspect: float, faces: list[Face]) -> tuple[int, int, int, int]:
    """Largest box with the given aspect that keeps the most faces.

    Only one axis is ever cropped: either left/right or top/bottom.
    """
    if width / height > aspect:
        cw = max(1, min(width, round(height * aspect)))
        x = _best_offset(width, cw, [(f.x, f.x + f.w) for f in faces])
        return x, 0, x + cw, height
    ch = max(1, min(height, round(width / aspect)))
    y = _best_offset(height, ch, [(f.y, f.y + f.h) for f in faces])
    return 0, y, width, y + ch
