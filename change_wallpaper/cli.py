import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from Xlib import display

from change_wallpaper.crop import (
    BACKGROUND_FACE_RATIO, crop_box, face_orientation, foreground_faces, is_square,
)
from change_wallpaper.debug import annotate
from change_wallpaper.faces import Face, FaceDetector
from change_wallpaper.log import get_logger, setup_logging
from change_wallpaper.monitors import Monitor, get_monitors
from change_wallpaper.xroot import set_root_background

Image.MAX_IMAGE_PIXELS = None

log = get_logger(__name__)

EXIF_ORIENTATION = 0x0112


@dataclass
class Candidate:
    path: Path
    width: int
    height: int

    @property
    def square(self) -> bool:
        return is_square(self.width, self.height)


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
    def __init__(self, background_ratio: float = BACKGROUND_FACE_RATIO):
        self.background_ratio = background_ratio
        self._detector: FaceDetector | None = None
        self._faces: dict[Path, list[Face]] = {}

    def load(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")

    def faces(self, path: Path, img: Image.Image | None = None) -> list[Face]:
        if path not in self._faces:
            if self._detector is None:
                self._detector = FaceDetector()
            self._faces[path] = self._detector.detect(img if img is not None else self.load(path))
            log.debug("%s: %d face(s)", path, len(self._faces[path]))
        return self._faces[path]

    def choose(self, monitor: Monitor, pools: dict[str, list[Candidate]], used: set[Path]) -> Candidate | None:
        """Pick a random image for the monitor.

        Square images from the other orientation are also eligible when their
        face layout prefers this monitor's orientation, and square images of
        this orientation are skipped when faces clearly prefer the other one.
        """
        own = monitor.orientation
        other = "horizontal" if own == "vertical" else "vertical"
        options = [(c, True) for c in pools[own]] + [(c, False) for c in pools[other] if c.square]
        random.shuffle(options)
        # prefer images not yet shown on another monitor
        options.sort(key=lambda o: o[0].path in used)
        fallback = None
        for cand, native in options:
            if not cand.square:
                return cand
            pref = face_orientation(foreground_faces(self.faces(cand.path), self.background_ratio))
            if pref == own or (pref is None and native):
                return cand
            if native and fallback is None:
                fallback = cand
        return fallback

    def render(self, path: Path, monitor: Monitor, debug: bool = False) -> tuple[Image.Image, Image.Image | None]:
        """Cropped image for the monitor and, if requested, the annotated source image."""
        img = self.load(path)
        faces = self.faces(path, img)
        kept = foreground_faces(faces, self.background_ratio)
        box = crop_box(img.width, img.height, monitor.aspect, kept)
        log.debug("%s: %s %s -> crop %s", monitor.name, path, img.size, box)
        cropped = img.crop(box).resize((monitor.width, monitor.height), Image.Resampling.LANCZOS)
        return cropped, annotate(img, monitor, box, kept, [f for f in faces if f not in kept]) if debug else None


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
    ap.add_argument("-V", "--vertical", nargs="+", default=[], metavar="PATH",
                    help="directories (recursive) or files for vertical monitors")
    ap.add_argument("-H", "--horizontal", nargs="+", default=[], metavar="PATH",
                    help="directories (recursive) or files for horizontal monitors")
    ap.add_argument("-o", "--override", metavar="IMAGE",
                    help="if this file exists, use it on all monitors instead of a random pick")
    ap.add_argument("-s", "--save", metavar="FILE", help="also save the combined image to FILE")
    ap.add_argument("-b", "--background-percent", type=percentage, default=BACKGROUND_FACE_RATIO * 100,
                    metavar="PERCENT",
                    help="ignore faces whose area is below PERCENT%% of the biggest face, 0-100 "
                         f"(default {BACKGROUND_FACE_RATIO * 100:g}, 0 disables)")
    ap.add_argument("-d", "--save-debug", metavar="FILE",
                    help="save the uncropped source images with crop area (green), faces (red) "
                         "and ignored background faces (yellow) to FILE")
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

    wp = Wallpaper(args.background_percent / 100)
    override = Path(args.override).expanduser() if args.override else None
    if override and override.is_file() and probe(override):
        log.info("Using override image %s", override)
        selection = {m: override for m in monitors}
    else:
        if override:
            log.debug("Override %s not usable, selecting randomly", override)
        start = time.perf_counter()
        pools = {"vertical": collect(args.vertical), "horizontal": collect(args.horizontal)}
        log.debug("Found %d vertical and %d horizontal image(s) in %.2fs",
                  len(pools["vertical"]), len(pools["horizontal"]), time.perf_counter() - start)
        start = time.perf_counter()
        selection = {}
        used: set[Path] = set()
        for m in monitors:
            cand = wp.choose(m, pools, used)
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
    dpy.close()
    return 0
