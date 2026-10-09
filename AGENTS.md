# AGENTS.md

Guidance for agents working on **change-wallpaper**: a Python CLI that sets a random,
face-aware wallpaper on every monitor of an X11/Xorg desktop.

## Project setup

- Dependencies are managed with **Poetry** (`pyproject.toml`). Never use bare `pip install`;
  add dependencies with `poetry add`.
- Python `>=3.11` (developed on 3.14). `numpy>=2.3` is required, since older versions have no wheels for 3.14.
- Runtime deps: `pillow`, `numpy`, `opencv-python-headless` (OpenCV 5 in use), `python-xlib`.
- Entry point: `poetry run change-wallpaper ...` (also `python -m change_wallpaper`).
- Everything must run **locally**. No external/cloud services. The only network access
  allowed is the one-time download of the YuNet and NanoDet-Plus models to `~/.cache/change-wallpaper/`.
- Face and person detection run on **CPU only**. CUDA was evaluated and rejected: PyPI OpenCV wheels have
  no CUDA, OpenCV 5's new graph engine ignores targets, and detection is not the bottleneck.

## Module layout (`change_wallpaper/`)

| Module | Responsibility |
|---|---|
| `cli.py` | Argument parsing, image collection/validation, selection, rendering, orchestration |
| `monitors.py` | Monitor discovery via RandR (`xrandr_get_monitors`), orientation |
| `detection.py` | `Box` (detected object: face or person), model download/cache, shared image helpers |
| `faces.py` | `FaceDetector` (OpenCV `FaceDetectorYN`, YuNet) |
| `people.py` | `PersonDetector` (NanoDet-Plus via `cv2.dnn`, own post-processing) |
| `crop.py` | Pure cropping logic: `Subjects`, crop box, background filter, square-image orientation |
| `debug.py` | Annotated debug image (source image + crop/face rectangles) |
| `xroot.py` | Setting the X root background pixmap |
| `log.py` | Logging setup and `get_logger()` |

Keep `crop.py` free of I/O so it stays easy to test with synthetic `Box` objects.
Detectors return `Box` objects; whether a box is a face or a person is given by where it is
stored (`Subjects.faces` / `Subjects.people`), not by its type.

## Command-line interface

| Option | Meaning |
|---|---|
| `-i/--images PATH...` | Files or directories (recursive), repeatable; images are sorted by orientation automatically |
| `-o/--override IMAGE` | If the file exists, use it on **all** monitors (full crop logic still applies); otherwise fall back to random selection |
| `-s/--save FILE` | Save the combined wallpaper |
| `-d/--save-debug FILE` | Save the debug image (uncropped sources with rectangles) |
| `-b/--background-percent PERCENT` | Background face/person threshold as a percentage, 0–100 (default `25`, `0` disables); converted to a 0–1 ratio internally |
| `-P/--no-people` | Disable the person-detection fallback |
| `-n/--dry-run` | Do not set the X root background |
| `-v/--verbose` / `-q/--silent` | DEBUG / WARNING log level (mutually exclusive; default INFO) |
| `-l/--log-file FILE` | Also log to a rotating file |

## Functional rules

### Monitors
- Read every monitor's size and position from RandR. A monitor is **vertical** when its height is greater than its width,
  otherwise **horizontal**. If RandR reports none, use the whole screen (with a warning).

### Image collection
- Files are added as is; directories are scanned **recursively**.
- Invalid images are discarded with a **warning**. Missing paths produce a warning.
- Validation must be **header-only** (`Image.verify()` + size + EXIF orientation tag 0x0112;
  swap width/height for orientations 5–8). **Never decode full images during scanning.**
  `exif_transpose` on every file made scans take ~28s for 575 files; header-only takes ~0.2s.
- Full decoding (`exif_transpose` + RGB) happens only in `Wallpaper.load()` for chosen images.

### Selection
- There is a single image list (`-i`); separate horizontal/vertical options were removed on purpose.
  Each image is classified by its real size (after EXIF rotation): `horizontal`, `vertical` or
  `square` (`Candidate.orientation`). This also fixes misfiled images.
- One random image per monitor among the images matching its orientation, plus square images.
- Avoid repeating the same image on multiple monitors when possible.
- **Square images** (aspect within ±5% of 1:1, `SQUARE_TOLERANCE`): face layout decides the
  preferred orientation (`layout_orientation`: union span of foreground faces, or people when there are no faces;
  one side >1.2× the other). A square image
  is used if its faces prefer this monitor's orientation or show no preference; if faces clearly
  prefer the other orientation, it is only used as a fallback.

### Cropping (`crop.crop_box`)
- Crop to the monitor aspect ratio using the **maximum possible area**: crop **only one axis**
  (left/right *or* top/bottom), never both.
- Window placement, in priority order:
  1. Contain the **largest number of whole faces**, respecting image boundaries.
  2. Tie-break by total face area.
  3. Center on the **area-weighted center** of the contained faces (bigger, closer faces weigh more),
     clamped so every contained face stays inside the window.
- **People fallback**: only when no face at all is detected (and `-P` is not given), people are
  detected and used like faces (`crop_box(..., people=...)`). A person box is kept **whole** when it
  fits the crop window; otherwise its **top 25%** (`PERSON_HEAD_FRACTION`, head and shoulders) is
  used on the y axis, not the middle of the box. Any span still longer than the window is reduced
  to its center point.
- No faces or people: center the crop.
- **Background faces/people**: faces (or people) with area below `BACKGROUND_FACE_RATIO` (0.25, i.e. `-b 25`) × the biggest face
  area are ignored for cropping and orientation (`foreground`). Typical case: one subject in
  front of a crowd; at 25% only the front subject is kept, while 10% still kept some crowd faces.
- Keep `README.md` (options table) and this file in sync whenever CLI options change.

### Face and person detection
- Faces: YuNet (`face_detection_yunet_2023mar.onnx`), score threshold 0.7.
- People: NanoDet-Plus (`object_detection_nanodet_2022nov.onnx`, COCO class 0 "person"), score
  threshold 0.5, NMS 0.6, 416×416 letterboxed input (top-left anchored). Outputs are paired by
  shape (`(1, N, 80)` scores with `(1, N, 32)` box distributions, N = 52², 26², 13² for strides
  8/16/32), not by output order, since the order differs from the opencv_zoo demo.
- OpenCV 5 has no `HOGDescriptor`, so the classic HOG people detector is not available.
- Detect on a copy scaled to max 1280 px on the long side, then scale boxes back and clip to the image.
- Results are cached per path for the duration of a run.
- OpenCV native logging is set to ERROR to avoid noise.

### Composition and root background
- Canvas = X screen size; each monitor's crop is resized (LANCZOS) to the monitor size and pasted at its position.
- Root background is set like `hsetroot`/`feh`: create a pixmap, upload in chunks below the X max
  request size, set `_XROOTPMAP_ID` and `ESETROOT_PMAP_ID`, set the window background,
  `RetainPermanent`, and free the previous pixmap (via `kill_client` on the old resource) when both
  properties match.

### Debug image
- Same canvas/layout as the wallpaper; each **full** source image is fitted (letterboxed) into its monitor area.
- Colors: crop area **green**, kept faces **red**, ignored background faces **yellow**,
  kept people **blue**, ignored background people **magenta**.

## Logging standards

- **No `print`.** All output goes through `logging`.
- Each module gets its own logger: `log = get_logger(__name__)`, which yields `change-wallpaper.<module>`.
- Format: `<ISO 8601 datetime with ms and timezone> <LEVEL> <logger name>: <message>`,
  e.g. `2026-10-09T12:17:06.430+02:00 WARNING change-wallpaper.cli: ...`.
- Levels: default INFO, `--verbose` DEBUG, `--silent` WARNING.
- Use lazy `%` formatting (`log.info("x %s", v)`), not f-strings.
- Level guidelines:
  - DEBUG: faces, people (with timing), crop boxes, counts, timings (scan, selection, rendering, background set), pixmap cleanup.
  - INFO: monitors, chosen images, override in use, model download, saved files, background set.
  - WARNING: invalid/missing files, monitor without image, no RandR monitors, interruption.
  - ERROR: nothing to set; unexpected exceptions are logged with `log.exception` (exit code 1; Ctrl+C exits 130).
- Rotating file logging: `RotatingFileHandler`, 10 MB per file, 5 files total (current + 4 backups).

## Coding conventions

- Match the existing style: type hints, dataclasses, small focused functions, module constants in UPPER_CASE.
- Sparse comments, only for non-obvious behavior.
- Keep performance in mind: the program runs often (e.g. on a timer). Watch for full image decodes and
  unnecessary face detection. Use the DEBUG timing logs to check.

## Testing / verification

- Use `-n` (dry run) with `-s`/`-d` to inspect output without changing the desktop.
- Use the `-d` debug image to visually verify crop and face decisions.
- Pure crop logic can be checked with synthetic faces, e.g.
  `crop_box(1000, 1000, 9/16, [Box(...)])`.
- Test with real photo folders (several hundred images) to catch performance regressions, and with
  photos of a subject in front of a crowd to check the background-face filter.
