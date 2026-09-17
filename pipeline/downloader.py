"""yt-dlp wrapper: metadata, subtitles, audio, and full video download.

All network work is shelled out to yt-dlp. Payloads land in
``<data_dir>/<video_id>/`` and the returned :class:`Video` points at them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.models import Video

DEFAULT_INFO_TIMEOUT = 180
DEFAULT_SUB_TIMEOUT = 300
DEFAULT_AUDIO_TIMEOUT = 1800
DEFAULT_VIDEO_TIMEOUT = 7200

VIDEO_FORMAT = "bv*[height<=1080]+ba/b[height<=1080]/b"


class DownloaderError(RuntimeError):
    pass


def ytdlp_command() -> str:
    """Prefer a yt-dlp installed next to the running interpreter (venv)."""
    sibling = Path(sys.executable).parent / "yt-dlp"
    if sibling.exists():
        return str(sibling)
    return "yt-dlp"


def run_yt_dlp(
    args: list[str],
    *,
    timeout: int = DEFAULT_INFO_TIMEOUT,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [ytdlp_command(), "--no-warnings", "--no-playlist", *args]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except FileNotFoundError as exc:
        raise DownloaderError("yt-dlp is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloaderError(f"yt-dlp timed out after {timeout}s") from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise DownloaderError(detail[-2000:] or f"yt-dlp exited {proc.returncode}")
    return proc


def fetch_info(url: str, *, timeout: int = DEFAULT_INFO_TIMEOUT) -> dict[str, Any]:
    proc = run_yt_dlp(["-J", url], timeout=timeout)
    line = next((ln for ln in proc.stdout.splitlines() if ln.strip()), "")
    if not line:
        raise DownloaderError("yt-dlp returned no metadata")
    try:
        info = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DownloaderError("could not parse yt-dlp metadata JSON") from exc
    if not isinstance(info, dict):
        raise DownloaderError("unexpected yt-dlp metadata payload")
    if info.get("_type") == "playlist":
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise DownloaderError("playlist has no downloadable entries")
        info = entries[0]
    return info


def video_from_info(info: dict[str, Any]) -> Video:
    duration = info.get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    video_id = str(info.get("id") or "unknown")
    url = info.get("webpage_url") or info.get("url") or ""
    return Video(
        video_id=video_id,
        url=url,
        title=info.get("title") or "Untitled",
        author=info.get("uploader") or info.get("channel") or "Unknown",
        duration=duration,
        description=info.get("description") or "",
        thumbnail=info.get("thumbnail"),
        webpage_url=url,
    )


def available_subtitle_langs(info: dict[str, Any]) -> list[str]:
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    langs: list[str] = []
    for source in (manual, auto):
        for lang in source:
            if lang not in langs:
                langs.append(lang)
    langs.sort(key=lambda x: (not x.startswith("en"), x))
    return langs


def choose_subtitle_lang(info: dict[str, Any], preferred: str = "") -> str | None:
    manual = set((info.get("subtitles") or {}).keys())
    auto = set((info.get("automatic_captions") or {}).keys())
    preferred = (preferred or "").strip().lower()
    if preferred:
        for candidate in (preferred, preferred.split("-")[0]):
            if candidate in manual:
                return candidate
        for candidate in (preferred, preferred.split("-")[0]):
            if candidate in auto:
                return candidate
    for candidate in ("en", "en-US", "en-GB"):
        if candidate in manual:
            return candidate
    if manual:
        return min(manual)
    for candidate in ("en", "en-US", "en-GB"):
        if candidate in auto:
            return candidate
    if auto:
        return min(auto)
    return None


def download_subtitles(
    url: str,
    out_dir: Path,
    *,
    video_id: str,
    lang: str,
    include_auto: bool = True,
    timeout: int = DEFAULT_SUB_TIMEOUT,
) -> str | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / f"{video_id}.%(ext)s")
    args = [
        "--skip-download",
        "--write-subs",
        "--sub-langs", lang,
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "-o", template,
    ]
    if include_auto:
        args.insert(1, "--write-auto-subs")
    run_yt_dlp(args + [url], timeout=timeout, check=False)
    return find_subtitle_file(out_dir, video_id)


def find_subtitle_file(out_dir: Path, video_id: str) -> str | None:
    if not out_dir.exists():
        return None
    candidates = sorted(out_dir.glob(f"{video_id}*.vtt"))
    return str(candidates[0]) if candidates else None


def _capture_after_move(
    args: list[str],
    out_dir: Path,
    *,
    timeout: int,
    fallback_globs: Iterable[str],
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = run_yt_dlp(args, timeout=timeout)
    for line in reversed(proc.stdout.splitlines()):
        candidate = line.strip()
        if candidate and os.path.isfile(candidate):
            return candidate
    for pattern in fallback_globs:
        hits = sorted(
            p for p in out_dir.glob(pattern)
            if p.is_file() and not p.name.endswith((".part", ".ytdl"))
        )
        if hits:
            return str(hits[-1])
    raise DownloaderError("yt-dlp reported success but no output file was found")


def download_audio(
    url: str,
    out_dir: Path,
    *,
    video_id: str,
    ext: str = "mp3",
    timeout: int = DEFAULT_AUDIO_TIMEOUT,
) -> str:
    template = str(out_dir / f"{video_id}.%(ext)s")
    return _capture_after_move(
        [
            "-x",
            "--audio-format", ext,
            "--audio-quality", "5",
            "-o", template,
            "--print", "after_move:filepath",
            url,
        ],
        out_dir,
        timeout=timeout,
        fallback_globs=[f"{video_id}.{ext}", f"{video_id}.*"],
    )


def download_video(
    url: str,
    out_dir: Path,
    *,
    video_id: str,
    max_height: int = 1080,
    fmt: str = VIDEO_FORMAT,
    timeout: int = DEFAULT_VIDEO_TIMEOUT,
) -> str:
    template = str(out_dir / f"{video_id}.%(ext)s")
    return _capture_after_move(
        [
            "-f", fmt,
            "--merge-output-format", "mp4",
            "-o", template,
            "--print", "after_move:filepath",
            url,
        ],
        out_dir,
        timeout=timeout,
        fallback_globs=["*.mp4", "*.mkv", "*.webm"],
    )


@dataclass
class DownloadPlan:
    want_audio: bool = True
    want_video: bool = True
    want_subtitles: bool = True
    language: str = ""
    max_height: int = 1080


def fetch(
    url: str,
    out_dir: str | Path = "data",
    *,
    plan: DownloadPlan | None = None,
    progress: Callable[[str], None] | None = None,
) -> Video:
    plan = plan or DownloadPlan()
    out_root = Path(out_dir)
    notify = progress or (lambda _msg: None)

    notify("fetching metadata")
    info = fetch_info(url)
    video = video_from_info(info)
    if not video.url:
        video.url = url
        video.webpage_url = url

    vdir = out_root / video.video_id
    vdir.mkdir(parents=True, exist_ok=True)

    if plan.want_subtitles:
        lang = choose_subtitle_lang(info, plan.language)
        if lang:
            notify(f"downloading subtitles ({lang})")
            video.subtitle_path = download_subtitles(
                video.url, vdir, video_id=video.video_id, lang=lang,
            )
        else:
            notify("no subtitles available")

    if plan.want_audio:
        notify("downloading audio")
        video.audio_path = download_audio(video.url, vdir, video_id=video.video_id)

    if plan.want_video:
        notify("downloading video")
        video.video_path = download_video(
            video.url, vdir, video_id=video.video_id, max_height=plan.max_height,
        )

    (vdir / "metadata.json").write_text(json.dumps(video.to_dict(), indent=2) + "\n")
    return video
