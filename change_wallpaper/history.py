# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Danilo Treffiletti

import json
import os
from pathlib import Path

from change_wallpaper.detection import MODEL_DIR
from change_wallpaper.log import get_logger

log = get_logger(__name__)

HISTORY_FILE = MODEL_DIR / "history.json"
HISTORY_SIZE = 5

Group = dict[str, str]


def load_history(path: Path = HISTORY_FILE) -> list[Group]:
    """Recent selections (monitor name -> image path), oldest first."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        log.warning("Ignoring unreadable history %s: %s", path, e)
        return []
    if not isinstance(data, list):
        log.warning("Ignoring malformed history %s", path)
        return []
    return [g for g in data if isinstance(g, dict)][-HISTORY_SIZE:]


def recent_paths(groups: list[Group]) -> set[Path]:
    return {Path(p) for g in groups for p in g.values() if isinstance(p, str)}


def save_history(groups: list[Group], group: Group, path: Path = HISTORY_FILE) -> None:
    """Append a group, keeping only the newest HISTORY_SIZE ones."""
    data = (groups + [group])[-HISTORY_SIZE:]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)
    log.debug("History saved to %s (%d group(s))", path, len(data))
