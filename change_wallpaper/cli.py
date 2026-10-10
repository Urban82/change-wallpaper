# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Danilo Treffiletti

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from Xlib import display

from change_wallpaper.crop import (
    BACKGROUND_FACE_RATIO, Subjects, crop_box, foreground, is_square, layout_orientation,
)
from change_wallpaper.debug import annotate
from change_wallpaper.detection import Box
from change_wallpaper.faces import FaceDetector
from change_wallpaper.history import load_history, recent_paths, save_history
from change_wallpaper.log import get_logger, setup_logging
from change_wallpaper.monitors import Monitor, get_monitors
from change_wallpaper.objects import ObjectDetector
from change_wallpaper.xroot import set_root_background

Image.MAX_IMAGE_PIXELS = None

log = get_logger(__name__)

EXIF_ORIENTATION = 0x0112

# option value -> subject kind, in default fallback order
FALLBACK_NAMES = {"people": "person", "animals": "animal", "vehicles": "vehicle"}
DEFAULT_FALLBACKS = tuple(FALLBACK_NAMES.values())


@dataclass
class Candidate:
    path: Path
    width: int
    height: int

    @property
    def square(self) -> bool:
        return is_square(self.width, self.height)

    @property
    def orientation(self) -> str:
        """'horizontal', 'vertical' or 'square' (usable for both)."""
        if self.square:
            return "square"
        return "vertical" if self.height > self.width else "horizontal"


def probe(path: Path) -> Candidate | None:
    try:
        with Image.open(path) as img:
            img.verify()
        # Header-only read: size plus EXIF orientation, without decoding pixels
        with Image.open(path) as img:
            width, height = img.size
            if img.getexif().get(EXIF_ORIENTATION, 1) in (5, 6, 7, 8):
                width, height = height, width
            return Candidate(path, width, height)
    except Exception as exc:
        log.warning("Discarding %s: not a valid image (%s: %s)", path, exc.__class__.__name__, exc)
        return None


def collect(entries: list[str]) -> list[Candidate]:
    seen: set[Path] = set()
    result = []
    for entry in entries:
        p = Path(entry).expanduser()
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            files = [p]
        else:
            log.warning("%s does not exist", p)
            continue
        for f in files:
            key = f.resolve()
            if key in seen:
                continue
            seen.add(key)
            if (c := probe(f)) is not None:
                result.append(c)
    return result


class Wallpaper:
    def __init__(self, background_ratio: float = BACKGROUND_FACE_RATIO,
                 fallbacks: tuple[str, ...] = DEFAULT_FALLBACKS):
        self.background_ratio = background_ratio
        self.fallbacks = fallbacks
        self._face_detector: FaceDetector | None = None
        self._object_detector: ObjectDetector | None = None
        self._subjects: dict[Path, Subjects] = {}

    def load(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")

    def _subjects_of(self, kind: str, boxes: list[Box]) -> Subjects:
        kept = foreground(boxes, self.background_ratio, kind)
        return Subjects(kind, kept, [b for b in boxes if b not in kept])

    def subjects(self, path: Path, img: Image.Image | None = None) -> Subjects:
        """Faces; when none, the first of people/animals/vehicles found (cached per path)."""
        if path in self._subjects:
            return self._subjects[path]
        if img is None:
            img = self.load(path)
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        start = time.perf_counter()
        faces = self._face_detector.detect(img)
        log.debug("%s: %d face(s) found in %.2fs", path, len(faces), time.perf_counter() - start)
        result = Subjects()
        if faces:
            result = self._subjects_of("face", faces)
        elif self.fallbacks:
            if self._object_detector is None:
                self._object_detector = ObjectDetector()
            start = time.perf_counter()
            found = self._object_detector.detect(img, self.fallbacks)
            log.debug("%s: no faces, found %s in %.2fs", path,
                      ", ".join(f"{len(found[k])} {k}(s)" for k in self.fallbacks), time.perf_counter() - start)
            for kind in self.fallbacks:
                if found[kind]:
                    log.debug("Detected %s: %s", kind, found[kind])
                    result = self._subjects_of(kind, found[kind])
                    break
        self._subjects[path] = result
        return result

    def choose(self, monitor: Monitor, images: list[Candidate], used: set[Path],
               recent: set[Path] = frozenset()) -> Candidate | None:
        """Pick a random image matching the monitor orientation.

        Square images are eligible for both orientations, unless the layout of
        their subjects clearly prefers the other one (then only used as a fallback).
        """
        own = monitor.orientation
        options = [c for c in images if c.orientation in (own, "square")]
        random.shuffle(options)
        # prefer images not yet shown on another monitor, then not shown in recent runs
        options.sort(key=lambda c: (c.path in used, c.path.resolve() in recent))
        fallback = None
        for cand in options:
            if not cand.square:
                return cand
            pref = layout_orientation(self.subjects(cand.path).main)
            if pref in (own, None):
                return cand
            if fallback is None:
                fallback = cand
        return fallback

    def render(self, path: Path, monitor: Monitor, debug: bool = False) -> tuple[Image.Image, Image.Image | None]:
        """Cropped image for the monitor and, if requested, the annotated source image."""
        img = self.load(path)
        found = self.subjects(path, img)
        box = crop_box(img.width, img.height, monitor.aspect, found.main, top_fallback=found.kind == "person")
        log.debug("%s: %s %s -> crop %s (subject: %s)", monitor.name, path, img.size, box, found.kind or "none")
        cropped = img.crop(box).resize((monitor.width, monitor.height), Image.Resampling.LANCZOS)
        return cropped, annotate(img, monitor, box, found) if debug else None


def fallbacks(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "none":
        return ()
    kinds = []
    for name in (v.strip().lower() for v in value.split(",") if v.strip()):
        if name not in FALLBACK_NAMES:
            raise argparse.ArgumentTypeError(
                f"unknown kind {name!r}, use {', '.join(FALLBACK_NAMES)} or 'none'")
        if FALLBACK_NAMES[name] not in kinds:
            kinds.append(FALLBACK_NAMES[name])
    return tuple(kinds)


def percentage(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from None
    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError(f"must be between 0 and 100, got {value}")
    return number


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Set a face-aware random wallpaper on every X11 monitor.")
    ap.add_argument("-i", "--images", nargs="+", action="extend", default=[], metavar="PATH",
                    help="image files or directories (recursive); sorted by orientation automatically")
    ap.add_argument("-o", "--override", metavar="IMAGE",
                    help="if this file exists, use it on all monitors instead of a random pick")
    ap.add_argument("-s", "--save", metavar="FILE", help="also save the combined image to FILE")
    ap.add_argument("-b", "--background-percent", type=percentage, default=BACKGROUND_FACE_RATIO * 100,
                    metavar="PERCENT",
                    help="ignore faces whose area is below PERCENT%% of the biggest face, 0-100 "
                         f"(default {BACKGROUND_FACE_RATIO * 100:g}, 0 disables)")
    ap.add_argument("-F", "--fallback", type=fallbacks, default=DEFAULT_FALLBACKS, metavar="KINDS",
                    help="what to look for, in order, when no face is found: comma separated list of "
                         f"{', '.join(FALLBACK_NAMES)}, or 'none' (default {','.join(FALLBACK_NAMES)})")
    ap.add_argument("-d", "--save-debug", metavar="FILE",
                    help="save the uncropped source images with the crop area (green) and the detected faces (red), "
                         "people (blue), animals (orange) or vehicles (magenta) to FILE; "
                         "ignored background ones are dashed")
    ap.add_argument("-n", "--dry-run", action="store_true", help="do not set the X root background")
    verbosity = ap.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="log DEBUG messages")
    verbosity.add_argument("-q", "--silent", action="store_true", help="log only warnings and errors")
    ap.add_argument("-l", "--log-file", metavar="FILE",
                    help="also log to FILE, rotated at 10MB keeping 5 files")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.silent else logging.INFO
    setup_logging(level, args.log_file)
    try:
        return run(args)
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130
    except Exception:
        log.exception("Unexpected error")
        return 1


def run(args: argparse.Namespace) -> int:
    dpy = display.Display()
    screen = dpy.screen()
    monitors = get_monitors(dpy)

    wp = Wallpaper(args.background_percent / 100, args.fallback)
    override = Path(args.override).expanduser() if args.override else None
    history = None
    if override and override.is_file() and probe(override):
        log.info("Using override image %s", override)
        selection = {m: override for m in monitors}
    else:
        history = load_history()
        recent = recent_paths(history)
        log.debug("%d recently used image(s) in history", len(recent))
        if override:
            log.debug("Override %s not usable, selecting randomly", override)
        start = time.perf_counter()
        images = collect(args.images)
        counts = {o: sum(c.orientation == o for c in images) for o in ("horizontal", "vertical", "square")}
        log.debug("Found %d horizontal, %d vertical and %d square image(s) in %.2fs",
                  counts["horizontal"], counts["vertical"], counts["square"], time.perf_counter() - start)
        start = time.perf_counter()
        selection = {}
        used: set[Path] = set()
        for m in monitors:
            cand = wp.choose(m, images, used, recent)
            if cand is None:
                log.warning("No image available for %s monitor %s", m.orientation, m.name)
                continue
            used.add(cand.path)
            selection[m] = cand.path
            log.info("Monitor %s: %s", m.name, cand.path)
        log.debug("Selection took %.2fs", time.perf_counter() - start)

    if not selection:
        log.error("No images to set")
        return 1

    start = time.perf_counter()
    canvas = Image.new("RGB", (screen.width_in_pixels, screen.height_in_pixels))
    debug_canvas = Image.new("RGB", canvas.size) if args.save_debug else None
    for m, path in selection.items():
        cropped, annotated = wp.render(path, m, debug_canvas is not None)
        canvas.paste(cropped, (m.x, m.y))
        if debug_canvas is not None:
            debug_canvas.paste(annotated, (m.x, m.y))
    log.debug("Rendering took %.2fs", time.perf_counter() - start)

    if args.save:
        canvas.save(args.save)
        log.info("Combined image saved to %s", args.save)
    if debug_canvas is not None:
        debug_canvas.save(args.save_debug)
        log.info("Debug image saved to %s", args.save_debug)
    if not args.dry_run:
        start = time.perf_counter()
        set_root_background(dpy, canvas)
        log.debug("Setting background took %.2fs", time.perf_counter() - start)
        if history is not None:
            save_history(history, {m.name: str(p.resolve()) for m, p in selection.items()})
    dpy.close()
    return 0
