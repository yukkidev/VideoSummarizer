"""Segment-aware transcript chunking with overlap and timestamp ranges."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pipeline.models import Chunk, Segment, Transcript

VTT_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEWLINE_RE = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    text = VTT_TAG_RE.sub(" ", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = MULTISPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTINEWLINE_RE.sub("\n", text)
    return text.strip()


def _join(segments: Sequence[Segment]) -> str:
    return clean_text(" ".join(s.text for s in segments if s.text.strip()))


def chunk_segments(
    segments: Iterable[Segment],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    segs = [s for s in segments if clean_text(s.text)]
    if not segs:
        return []

    chunks: list[Chunk] = []
    current: list[Segment] = []
    current_len = 0

    def emit() -> None:
        nonlocal current, current_len
        text = _join(current)
        if text:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    start=current[0].start,
                    end=current[-1].end,
                    text=text,
                )
            )
        overlap: list[Segment] = []
        overlap_len = 0
        for seg in reversed(current):
            seg_len = len(clean_text(seg.text)) + 1
            if overlap_len + seg_len > overlap_chars:
                break
            overlap.insert(0, seg)
            overlap_len += seg_len
        current = overlap
        current_len = overlap_len

    for seg in segs:
        seg_len = len(clean_text(seg.text)) + 1
        if current and current_len + seg_len > max_chars:
            emit()
            while current and current_len + seg_len > max_chars:
                current.pop(0)
                current_len = sum(len(clean_text(s.text)) + 1 for s in current)
        current.append(seg)
        current_len += seg_len
    if current:
        text = _join(current)
        if text:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    start=current[0].start,
                    end=current[-1].end,
                    text=text,
                )
            )
    return chunks


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    text = clean_text(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(
                sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)
            )

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def emit() -> None:
        nonlocal current, current_len
        body = " ".join(current).strip()
        if body:
            chunks.append(Chunk(index=len(chunks), start=0.0, end=0.0, text=body))
        overlap: list[str] = []
        overlap_len = 0
        for unit in reversed(current):
            if overlap_len + len(unit) + 1 > overlap_chars:
                break
            overlap.insert(0, unit)
            overlap_len += len(unit) + 1
        current = overlap
        current_len = overlap_len

    for unit in units:
        if current and current_len + len(unit) + 1 > max_chars:
            emit()
        current.append(unit)
        current_len += len(unit) + 1
    if current:
        body = " ".join(current).strip()
        if body:
            chunks.append(Chunk(index=len(chunks), start=0.0, end=0.0, text=body))
    return chunks


def chunks_from_transcript(
    transcript: Transcript,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    if transcript.segments:
        return chunk_segments(
            transcript.segments, max_chars=max_chars, overlap_chars=overlap_chars,
        )
    return chunk_text(
        transcript.text, max_chars=max_chars, overlap_chars=overlap_chars,
    )


def timestamp_range(chunk: Chunk) -> str | None:
    from pipeline.models import format_timestamp

    if chunk.start == 0.0 and chunk.end == 0.0:
        return None
    return f"{format_timestamp(chunk.start)}-{format_timestamp(chunk.end)}"
