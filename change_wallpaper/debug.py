from PIL import Image, ImageDraw

from change_wallpaper.crop import Subjects
from change_wallpaper.log import get_logger
from change_wallpaper.monitors import Monitor

log = get_logger(__name__)

Color = tuple[int, int, int]

CROP_COLOR: Color = (0, 255, 0)
SUBJECT_COLORS: dict[str, Color] = {
    "face": (255, 0, 0),
    "person": (0, 128, 255),
    "animal": (255, 140, 0),
    "vehicle": (255, 0, 255),
}
# Background (ignored) subjects: same hue, darker, dashed
BACKGROUND_SHADE = 0.55


def _darker(color: Color) -> Color:
    return tuple(round(c * BACKGROUND_SHADE) for c in color)


def _dashed_rectangle(draw: ImageDraw.ImageDraw, r: list[int], color: Color, width: int) -> None:
    x0, y0, x1, y1 = r
    # shorter dashes on small boxes so they stay visible
    dash = max(width, min(width * 4, (min(x1 - x0, y1 - y0)) // 4))
    gap = max(width, dash * 3 // 4)
    for a, b, horizontal, fixed in ((x0, x1, True, y0), (x0, x1, True, y1),
                                    (y0, y1, False, x0), (y0, y1, False, x1)):
        for start in range(a, b + 1, dash + gap):
            end = min(start + dash, b)
            if horizontal:
                draw.line([(start, fixed), (end, fixed)], fill=color, width=width)
            else:
                draw.line([(fixed, start), (fixed, end)], fill=color, width=width)


def annotate(img: Image.Image, monitor: Monitor, box: tuple[int, int, int, int],
             found: Subjects) -> Image.Image:
    """Whole source image fitted into the monitor area, with crop and subject rectangles.

    Background (ignored) subjects use a darker shade of the same color, dashed.
    """
    scale = min(monitor.width / img.width, monitor.height / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    shown = img.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(shown)
    line = max(2, round(min(monitor.width, monitor.height) / 300))

    def scaled(x0: float, y0: float, x1: float, y1: float) -> list[int]:
        return [round(x0 * scale), round(y0 * scale), round(x1 * scale) - 1, round(y1 * scale) - 1]

    if found.kind is not None:
        color = SUBJECT_COLORS[found.kind]
        for b in found.background:
            _dashed_rectangle(draw, scaled(b.x, b.y, b.x + b.w, b.y + b.h), _darker(color), line)
        for b in found.main:
            draw.rectangle(scaled(b.x, b.y, b.x + b.w, b.y + b.h), outline=color, width=line)
    draw.rectangle(scaled(*box), outline=CROP_COLOR, width=line)

    cell = Image.new("RGB", (monitor.width, monitor.height))
    cell.paste(shown, ((monitor.width - size[0]) // 2, (monitor.height - size[1]) // 2))
    return cell
