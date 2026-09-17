"""Transcription backends: faster-whisper (preferred) and subtitle parsing."""

from __future__ import annotations

import html
import importlib.util
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from pipeline.models import Segment, Transcript

ProgressFn = Callable[[str, float], None]

DEFAULT_WHISPER_MODEL = "small"
TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


class TranscriptionError(RuntimeError):
    pass


def faster_whisper_available() -> bool:
    if "faster_whisper" in sys.modules:
        return sys.modules["faster_whisper"] is not None
    return importlib.util.find_spec("faster_whisper") is not None


def whisper_model_name(model: str | None = None) -> str:
    return model or os.environ.get("VS_WHISPER_MODEL") or DEFAULT_WHISPER_MODEL


def _seconds(groups: tuple[str, ...]) -> float:
    h, m, s, ms = (int(g) for g in groups)
    return h * 3600 + m * 60 + s + ms / 1000.0


def clean_caption_text(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def parse_caption_file(path: str | Path) -> list[Segment]:
    lines = Path(path).read_text(errors="replace").splitlines()
    segments: list[Segment] = []
    i = 0
    while i < len(lines):
        match = TIMESTAMP_RE.search(lines[i])
        if not match:
            i += 1
            continue
        start = _seconds(match.groups()[:4])
        end = _seconds(match.groups()[4:])
        i += 1
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if TIMESTAMP_RE.search(line):
                break
            if line.strip() == "":
                i += 1
                if body:
                    break
                continue
            body.append(line)
            i += 1
        text = clean_caption_text(" ".join(body))
        if not text:
            continue
        if segments:
            prev = segments[-1]
            if text == prev.text or prev.text.endswith(text):
                prev.end = max(prev.end, end)
                continue
            if text.startswith(prev.text):
                prev.text = text
                prev.end = max(prev.end, end)
                continue
        segments.append(Segment(start=start, end=end, text=text))
    return segments


def transcript_from_captions(path: str | Path, language: str = "") -> Transcript:
    segments = parse_caption_file(path)
    if not segments:
        raise TranscriptionError(f"no caption cues found in {path}")
    return Transcript(
        text=" ".join(s.text for s in segments),
        segments=segments,
        source=f"subtitles:{Path(path).suffix.lstrip('.') or 'vtt'}",
        language=language,
        duration=segments[-1].end,
    )


def transcribe_audio(
    audio_path: str | Path,
    *,
    model: str | None = None,
    device: str = "auto",
    language: str = "",
    compute_type: str | None = None,
    progress: ProgressFn | None = None,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Install it with "
            "`pip install faster-whisper` or `pip install -e .[whisper]`."
        ) from exc

    model_name = whisper_model_name(model)
    notify = progress or (lambda _m, _p: None)

    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    notify(f"loading whisper model '{model_name}' ({device}/{compute_type})", 0.0)
    whisper = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_iter, info = whisper.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=5,
        vad_filter=True,
    )
    total = float(info.duration or 0.0)
    segments: list[Segment] = []
    for raw in segments_iter:
        text = clean_caption_text(raw.text)
        if not text:
            continue
        segments.append(Segment(start=float(raw.start), end=float(raw.end), text=text))
        if total:
            notify("transcribing", min(0.99, float(raw.end) / total))
    if not segments:
        raise TranscriptionError("whisper produced no speech segments")
    notify("transcribing", 1.0)
    return Transcript(
        text=" ".join(s.text for s in segments),
        segments=segments,
        source=f"faster-whisper/{model_name}",
        language=getattr(info, "language", "") or "",
        duration=total or segments[-1].end,
    )


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 - any failure means no CUDA
        return False


def transcribe(
    *,
    audio_path: str | Path | None = None,
    subtitle_path: str | Path | None = None,
    model: str | None = None,
    device: str = "auto",
    language: str = "",
    progress: ProgressFn | None = None,
    force_subtitles: bool = False,
) -> Transcript:
    """Transcribe using the best available source.

    Audio plus faster-whisper wins; subtitles are the fallback (or the
    explicit choice when ``force_subtitles`` is set).
    """
    if subtitle_path and (force_subtitles or not audio_path or not faster_whisper_available()):
        return transcript_from_captions(subtitle_path, language)
    if audio_path:
        return transcribe_audio(
            audio_path, model=model, device=device, language=language,
            progress=progress,
        )
    if subtitle_path:
        return transcript_from_captions(subtitle_path, language)
    raise TranscriptionError("no audio or subtitle source available")
