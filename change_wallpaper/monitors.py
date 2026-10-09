# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Danilo Treffiletti

from dataclasses import dataclass

from Xlib import display

from change_wallpaper.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Monitor:
    name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def vertical(self) -> bool:
        return self.height > self.width

    @property
    def orientation(self) -> str:
        return "vertical" if self.vertical else "horizontal"

    @property
    def aspect(self) -> float:
        return self.width / self.height


def get_monitors(dpy: display.Display) -> list[Monitor]:
    root = dpy.screen().root
    reply = root.xrandr_get_monitors(is_active=True)
    monitors = [
        Monitor(
            name=dpy.get_atom_name(m.name),
            x=m.x,
            y=m.y,
            width=m.width_in_pixels,
            height=m.height_in_pixels,
        )
        for m in reply.monitors
    ]
    if not monitors:
        log.warning("RandR reported no monitors, using the whole screen")
        screen = dpy.screen()
        monitors = [Monitor("screen", 0, 0, screen.width_in_pixels, screen.height_in_pixels)]
    for m in monitors:
        log.info("Monitor %s: %dx%d+%d+%d %s", m.name, m.width, m.height, m.x, m.y, m.orientation)
    return monitors
