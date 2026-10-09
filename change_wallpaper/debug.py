from PIL import Image, ImageDraw

from change_wallpaper.crop import Subjects
from change_wallpaper.log import get_logger
from change_wallpaper.monitors import Monitor

log = get_logger(__name__)

CROP_COLOR = (0, 255, 0)
FACE_COLOR = (255, 0, 0)
BACKGROUND_FACE_COLOR = (255, 255, 0)
PERSON_COLOR = (0, 128, 255)
BACKGROUND_PERSON_COLOR = (255, 0, 255)


def annotate(img: Image.Image, monitor: Monitor, box: tuple[int, int, int, int],
             found: Subjects) -> Image.Image:
    """Whole source image fitted into the monitor area, with crop, face and person rectangles."""
    scale = min(monitor.width / img.width, monitor.height / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    shown = img.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(shown)
    line = max(2, round(min(monitor.width, monitor.height) / 300))

    def rect(x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
        r = [round(x0 * scale), round(y0 * scale), round(x1 * scale) - 1, round(y1 * scale) - 1]
        draw.rectangle(r, outline=color, width=line)

    for boxes, color in ((found.background_people, BACKGROUND_PERSON_COLOR),
                         (found.people, PERSON_COLOR),
                         (found.background_faces, BACKGROUND_FACE_COLOR),
                         (found.faces, FACE_COLOR)):
        for f in boxes:
            rect(f.x, f.y, f.x + f.w, f.y + f.h, color)
    rect(*box, CROP_COLOR)

    cell = Image.new("RGB", (monitor.width, monitor.height))
    cell.paste(shown, ((monitor.width - size[0]) // 2, (monitor.height - size[1]) // 2))
    return cell
