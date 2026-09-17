"""Best-effort desktop notifications via `notify-send` (Linux).

Used by background jobs so the user can walk away while long downloads and
transcriptions run. Silently does nothing when the tool is unavailable or
notifications are disabled in config.
"""

from __future__ import annotations

import shutil
import subprocess


def desktop_notify(title: str, body: str = "", *, enabled: bool = True) -> bool:
    if not enabled:
        return False
    sender = shutil.which("notify-send")
    if sender is None:
        return False
    try:
        subprocess.run(
            [sender, "--app-name=VideoSummarizer", title, body],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False
