# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Danilo Treffiletti

import cv2
import numpy as np
from PIL import Image

from change_wallpaper.detection import ZOO_URL, Box, clipped, ensure_model, quiet_opencv, to_bgr
from change_wallpaper.log import get_logger

log = get_logger(__name__)

MODEL_URL = ZOO_URL + "object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
INPUT_SIZE = 416
NUM_CLASSES = 80
REG_MAX = 7
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)

# COCO class ids grouped by subject kind
CLASS_GROUPS: dict[str, tuple[int, ...]] = {
    "person": (0,),
    # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    "animal": tuple(range(14, 24)),
    # bicycle, car, motorcycle, airplane, bus, train, truck, boat
    "vehicle": tuple(range(1, 9)),
}


class ObjectDetector:
    """NanoDet-Plus (COCO) used to find people (also seen from behind), animals and vehicles."""

    def __init__(self, score_threshold: float = 0.5, nms_threshold: float = 0.6):
        quiet_opencv()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._net = cv2.dnn.readNet(str(ensure_model(MODEL_URL, "NanoDet-Plus")))
        self._project = np.arange(REG_MAX + 1, dtype=np.float32)
        self._centers: dict[int, np.ndarray] = {}

    def _anchor_centers(self, count: int) -> tuple[np.ndarray, int]:
        side = int(round(count ** 0.5))
        stride = INPUT_SIZE // side
        if count not in self._centers:
            ys, xs = np.mgrid[0:side, 0:side]
            offset = 0.5 * (stride - 1)
            self._centers[count] = np.column_stack(
                (xs.ravel() * stride + offset, ys.ravel() * stride + offset)
            ).astype(np.float32)
        return self._centers[count], stride

    def detect(self, img: Image.Image, kinds: tuple[str, ...] = tuple(CLASS_GROUPS)) -> dict[str, list[Box]]:
        """Boxes for each requested kind, from a single network pass."""
        bgr, pre_scale = to_bgr(img)
        # letterbox into the square network input, anchored top-left
        fit = INPUT_SIZE / max(bgr.shape[:2])
        h, w = round(bgr.shape[0] * fit), round(bgr.shape[1] * fit)
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
        canvas[:h, :w] = cv2.resize(bgr, (w, h)).astype(np.float32)
        self._net.setInput(cv2.dnn.blobFromImage((canvas - MEAN) / STD))
        outs = self._net.forward(self._net.getUnconnectedOutLayersNames())

        # outputs come as (1, N, 80) class scores and (1, N, 32) box distributions
        # per stride; pair them by N instead of relying on output order
        scores_by_n = {o.shape[1]: o[0] for o in outs if o.shape[-1] == NUM_CLASSES}
        boxes_by_n = {o.shape[1]: o[0] for o in outs if o.shape[-1] == 4 * (REG_MAX + 1)}
        to_source = 1.0 / (fit * pre_scale)
        return {kind: self._group(img, scores_by_n, boxes_by_n, CLASS_GROUPS[kind], to_source)
                for kind in kinds}

    def _group(self, img: Image.Image, scores_by_n: dict[int, np.ndarray], boxes_by_n: dict[int, np.ndarray],
               classes: tuple[int, ...], to_source: float) -> list[Box]:
        boxes, scores = [], []
        for n, cls in scores_by_n.items():
            score = cls[:, list(classes)].max(axis=1)
            keep = score >= self.score_threshold
            if not keep.any():
                continue
            centers, stride = self._anchor_centers(n)
            reg = boxes_by_n[n][keep].reshape(-1, REG_MAX + 1)
            reg = np.exp(reg - reg.max(axis=1, keepdims=True))
            dist = (reg / reg.sum(axis=1, keepdims=True) @ self._project).reshape(-1, 4) * stride
            c = centers[keep]
            boxes.append(np.column_stack((c[:, 0] - dist[:, 0], c[:, 1] - dist[:, 1],
                                          dist[:, 0] + dist[:, 2], dist[:, 1] + dist[:, 3])))
            scores.append(score[keep])
        if not boxes:
            return []
        all_boxes = np.concatenate(boxes)
        all_scores = np.concatenate(scores)
        indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(),
                                   self.score_threshold, self.nms_threshold)
        found = []
        for i in np.array(indices).ravel():
            x, y, bw, bh = (float(v) * to_source for v in all_boxes[i])
            if (box := clipped(img, x, y, bw, bh)) is not None:
                found.append(box)
        return found
