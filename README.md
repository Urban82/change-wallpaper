# change-wallpaper

Sets a random, face-aware wallpaper on every X11/Xorg monitor.

The program reads all monitors (size, position, horizontal/vertical orientation) from Xorg/RandR,
sorts the given images by orientation (from their actual size), picks a random one matching each
monitor's orientation, crops it to the
monitor's aspect ratio keeping the detected faces (or people, animals or vehicles, when no face is visible), and sets the combined image as the root background.

## Installation

```sh
poetry install
```

The YuNet face detection model and the NanoDet-Plus object detection model are downloaded once to
`~/.cache/change-wallpaper/`.
Everything else runs locally, on the CPU.

## Usage

```sh
poetry run change-wallpaper -i /path/to/photos extra.jpg -o /path/to/override.jpg
```

## Options

| Option | Description |
|---|---|
| `-i`, `--images PATH [PATH ...]` | Image files or directories (scanned recursively). Can be repeated. Images are sorted automatically: wider than tall → horizontal monitors, taller than wide → vertical monitors, square → either. |
| `-o`, `--override IMAGE` | If this file exists, use it on every monitor instead of a random pick (cropping still applies). If it doesn't exist, images are picked randomly as usual. |
| `-s`, `--save FILE` | Also save the combined wallpaper to `FILE`. |
| `-d`, `--save-debug FILE` | Save a debug image: the uncropped source images in the monitor layout, with the crop area (green) and the detected faces (red), people (blue), animals (orange) or vehicles (magenta). Ignored background ones are drawn dashed, in a darker shade of the same color. |
| `-b`, `--background-percent PERCENT` | Ignore faces (or people, animals, vehicles) whose area is below `PERCENT`% of the biggest one, from 0 to 100. Default `25`; `0` disables the filter. |
| `-F`, `--fallback KINDS` | What to look for, in order, when no face is found: a comma-separated list of `people`, `animals`, `vehicles`, or `none` to disable. Default `people,animals,vehicles`. |
| `-n`, `--dry-run` | Do everything except setting the X root background (the selection history is not updated). |
| `-v`, `--verbose` | Show DEBUG messages (detected subjects, crop boxes, timings). |
| `-q`, `--silent` | Show only warnings and errors. |
| `-l`, `--log-file FILE` | Also log to `FILE`, rotated at 10 MB, keeping 5 files. |
| `-h`, `--help` | Show the help message. |

`-v` and `-q` cannot be used together. Without either, INFO messages are shown.

## How images are chosen and cropped

The paths of the images shown in the last 5 runs are kept in
`~/.cache/change-wallpaper/history.json`. Those images are avoided on every monitor while other
matching images are available. The history is not updated in dry-run mode or when the override
image is used.

- Invalid images are skipped with a warning.
- Orientation comes from the image size, taking the EXIF rotation into account.
- The same image is not repeated on multiple monitors when there are enough images.
- Each image is cropped to the monitor's aspect ratio along a single axis only (left/right or
  top/bottom), keeping the largest possible area.
- The crop keeps as many whole faces as possible and is centered on them, with bigger (closer)
  faces weighing more.
- When no face is found, the program looks for people (e.g. seen from behind), then animals,
  then vehicles (see `-F`), and uses the first kind found.
  - People are kept whole when they fit; otherwise the crop is centered on the top 25% of the
    person (head and shoulders) rather than the middle of the body.
  - Animals and vehicles are kept whole when they fit; otherwise the crop is centered on them.
- With nothing detected, the crop is centered.
- Small background subjects (see `-b`) are ignored.
- Square images can be used for either orientation when their faces are clearly arranged
  horizontally or vertically.

## Background

The background is set through `_XROOTPMAP_ID`/`ESETROOT_PMAP_ID`, like `hsetroot` and `feh`, so
transparent terminals and compositors pick it up.

## Examples

Check the result without changing the desktop:

```sh
poetry run change-wallpaper -n -s /tmp/wallpaper.jpg -d /tmp/debug.jpg -i /path/to/photos
```

Run quietly with a log file:

```sh
poetry run change-wallpaper -q -l /path/to/change-wallpaper.log -i /path/to/photos
```

## License

Copyright (C) 2026 Danilo Treffiletti

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program (see
[LICENSE](LICENSE)). If not, see <https://www.gnu.org/licenses/>.

The face and object detection models are downloaded at runtime and are not part of this project:
YuNet (MIT) and NanoDet-Plus (Apache-2.0), both from the OpenCV model zoo.
