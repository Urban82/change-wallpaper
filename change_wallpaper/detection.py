import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from change_wallpaper.log import get_logger

log = get_logger(__name__)

MODEL_DIR = Path.home() / ".cache" / "change-wallpaper"
ZOO_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/"
DETECT_MAX_SIDE = 1280


@dataclass(frozen=True)
class Box:
    """A detected object (face or person) in source image pixels."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    def __repr__(self) -> str:
        return f"Box(x={self.x:.0f}, y={self.y:.0f}, w={self.w:.0f}, h={self.h:.0f})"


def ensure_model(url: str, name: str) -> Path:
    """Download a model once and cache it locally."""
    path = MODEL_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s model from %s to %s", name, url, path)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def quiet_opencv() -> None:
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


def to_bgr(img: Image.Image, max_side: int = DETECT_MAX_SIDE) -> tuple[np.ndarray, float]:
    """BGR array of img scaled down to max_side, and the scale used."""
    scale = min(1.0, max_side / max(img.size))
    small = img
    if scale < 1.0:
        small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    return cv2.cvtColor(np.asarray(small.convert("RGB")), cv2.COLOR_RGB2BGR), scale


def clipped(img: Image.Image, x: float, y: float, w: float, h: float) -> Box | None:
    """Box clipped to the image bounds, or None if nothing is left."""
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(img.width), x + w), min(float(img.height), y + h)
    return Box(x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None
