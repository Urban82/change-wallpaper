# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Danilo Treffiletti

import cv2
from PIL import Image

from change_wallpaper.detection import ZOO_URL, Box, clipped, ensure_model, quiet_opencv, to_bgr
from change_wallpaper.log import get_logger

log = get_logger(__name__)

MODEL_URL = ZOO_URL + "face_detection_yunet/face_detection_yunet_2023mar.onnx"


class FaceDetector:
    def __init__(self, score_threshold: float = 0.7):
        quiet_opencv()
        self._detector = cv2.FaceDetectorYN.create(
            str(ensure_model(MODEL_URL, "YuNet")), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, img: Image.Image) -> list[Box]:
        bgr, scale = to_bgr(img)
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, found = self._detector.detect(bgr)
        if found is None:
            return []
        faces = []
        for row in found:
            if (face := clipped(img, *(float(v) / scale for v in row[:4]))) is not None:
                faces.append(face)
        log.debug("Detected faces: %s", faces)
        return faces
