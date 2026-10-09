# change-wallpaper

Sets a random, face-aware wallpaper on every X11/Xorg monitor.

```sh
poetry install
poetry run change-wallpaper -H ~/Pictures/landscape extra.jpg -V ~/Pictures/portrait -o ~/override.jpg
```

- `-H/--horizontal PATH...`, `-V/--vertical PATH...`: files or directories (scanned recursively). Invalid images are skipped with a warning.
- `-o/--override IMAGE`: if the file exists, it is used on every monitor instead of a random pick.
- `-s/--save FILE`: also save the combined image; `-n/--dry-run`: don't set the background; `-v`: verbose.

Each image is cropped to the monitor aspect ratio along a single axis only. The crop window keeps as many
faces (OpenCV YuNet, run locally) as possible, or is centered when there are no faces. Square images can be
used for either orientation when their face layout prefers it. The YuNet model is downloaded once to
`~/.cache/change-wallpaper/`. The background is set via `_XROOTPMAP_ID`/`ESETROOT_PMAP_ID`, like `hsetroot`.
