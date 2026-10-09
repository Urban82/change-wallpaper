import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from change_wallpaper.log import get_logger

log = get_logger(__name__)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_PATH = Path.home() / ".cache" / "change-wallpaper" / "face_detection_yunet_2023mar.onnx"
DETECT_MAX_SIDE = 1280


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    w: float
    h: float


def ensure_model() -> Path:
    """Download the YuNet model once and cache it locally."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading YuNet model from %s to %s", MODEL_URL, MODEL_PATH)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(MODEL_PATH)
    return MODEL_PATH


class FaceDetector:
    def __init__(self, score_threshold: float = 0.7):
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        self._detector = cv2.FaceDetectorYN.create(
            str(ensure_model()), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, img: Image.Image) -> list[Face]:
        scale = min(1.0, DETECT_MAX_SIDE / max(img.size))
        small = img
        if scale < 1.0:
            small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
        bgr = cv2.cvtColor(np.asarray(small.convert("RGB")), cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, found = self._detector.detect(bgr)
        if found is None:
            return []
        faces = []
        for row in found:
            x, y, w, h = (float(v) / scale for v in row[:4])
            # clip to image bounds
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(img.width), x + w), min(float(img.height), y + h)
            if x1 > x0 and y1 > y0:
                faces.append(Face(x0, y0, x1 - x0, y1 - y0))
        log.debug("Detected faces: %s", faces)
        return faces
