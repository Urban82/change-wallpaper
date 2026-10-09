from PIL import Image, ImageDraw

from change_wallpaper.faces import Face
from change_wallpaper.log import get_logger
from change_wallpaper.monitors import Monitor

log = get_logger(__name__)

CROP_COLOR = (0, 255, 0)
FACE_COLOR = (255, 0, 0)
BACKGROUND_FACE_COLOR = (255, 255, 0)


def annotate(img: Image.Image, monitor: Monitor, box: tuple[int, int, int, int],
             faces: list[Face], background: list[Face] = ()) -> Image.Image:
    """Whole source image fitted into the monitor area, with crop, face and background face rectangles."""
    scale = min(monitor.width / img.width, monitor.height / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    shown = img.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(shown)
    line = max(2, round(min(monitor.width, monitor.height) / 300))

    def rect(x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
        r = [round(x0 * scale), round(y0 * scale), round(x1 * scale) - 1, round(y1 * scale) - 1]
        draw.rectangle(r, outline=color, width=line)

    for f in background:
        rect(f.x, f.y, f.x + f.w, f.y + f.h, BACKGROUND_FACE_COLOR)
    for f in faces:
        rect(f.x, f.y, f.x + f.w, f.y + f.h, FACE_COLOR)
    rect(*box, CROP_COLOR)

    cell = Image.new("RGB", (monitor.width, monitor.height))
    cell.paste(shown, ((monitor.width - size[0]) // 2, (monitor.height - size[1]) // 2))
    return cell
