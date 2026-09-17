"""Per-video artifact persistence under data/<video_id>/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pipeline.models import Answer, Chunk, Summary, Transcript, Video

METADATA = "metadata.json"
TRANSCRIPT = "transcript.json"
CHUNKS = "chunks.json"
SUMMARY = "summary.json"
SUMMARY_MD = "summary.md"
VECTORS = "vectors.json"
ANSWERS = "answers.jsonl"


def video_dir(data_dir: str | Path, video_id: str) -> Path:
    return Path(data_dir) / video_id


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save_video(data_dir: str | Path, video: Video) -> Path:
    path = video_dir(data_dir, video.video_id) / METADATA
    _write_json(path, video.to_dict())
    return path


def load_video(data_dir: str | Path, video_id: str) -> Video | None:
    raw = _read_json(video_dir(data_dir, video_id) / METADATA)
    return Video.from_dict(raw) if isinstance(raw, dict) else None


def save_transcript(data_dir: str | Path, video_id: str, transcript: Transcript) -> Path:
    path = video_dir(data_dir, video_id) / TRANSCRIPT
    _write_json(path, transcript.to_dict())
    return path


def load_transcript(data_dir: str | Path, video_id: str) -> Transcript | None:
    raw = _read_json(video_dir(data_dir, video_id) / TRANSCRIPT)
    return Transcript.from_dict(raw) if isinstance(raw, dict) else None


def save_chunks(data_dir: str | Path, video_id: str, chunks: list[Chunk]) -> Path:
    path = video_dir(data_dir, video_id) / CHUNKS
    _write_json(path, [c.to_dict() for c in chunks])
    return path


def load_chunks(data_dir: str | Path, video_id: str) -> list[Chunk] | None:
    raw = _read_json(video_dir(data_dir, video_id) / CHUNKS)
    if not isinstance(raw, list):
        return None
    return [Chunk.from_dict(c) for c in raw]


def save_summary(data_dir: str | Path, video_id: str, summary: Summary) -> Path:
    path = video_dir(data_dir, video_id) / SUMMARY
    _write_json(path, summary.to_dict())
    return path


def load_summary(data_dir: str | Path, video_id: str) -> Summary | None:
    raw = _read_json(video_dir(data_dir, video_id) / SUMMARY)
    return Summary.from_dict(raw) if isinstance(raw, dict) else None


def save_summary_markdown(data_dir: str | Path, video_id: str, markdown: str) -> Path:
    path = video_dir(data_dir, video_id) / SUMMARY_MD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown)
    return path


def save_vectors(
    data_dir: str | Path,
    video_id: str,
    model: str,
    vectors: list[list[float]],
    chunk_count: int,
) -> Path:
    path = video_dir(data_dir, video_id) / VECTORS
    _write_json(
        path,
        {"model": model, "chunk_count": chunk_count, "vectors": vectors},
    )
    return path


def load_vectors(
    data_dir: str | Path, video_id: str,
) -> dict[str, Any] | None:
    raw = _read_json(video_dir(data_dir, video_id) / VECTORS)
    if not isinstance(raw, dict) or not isinstance(raw.get("vectors"), list):
        return None
    return raw


def append_answer(data_dir: str | Path, answer: Answer, created: str = "") -> Path:
    path = video_dir(data_dir, answer.video_id) / ANSWERS
    path.parent.mkdir(parents=True, exist_ok=True)
    record = answer.to_dict()
    record["created"] = created
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_answers(data_dir: str | Path, video_id: str) -> list[dict[str, Any]]:
    path = video_dir(data_dir, video_id) / ANSWERS
    if not path.exists():
        return []
    answers: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            answers.append(record)
    return answers


def video_record(data_dir: str | Path, video_id: str) -> dict[str, Any] | None:
    video = load_video(data_dir, video_id)
    if video is None:
        return None
    vdir = video_dir(data_dir, video_id)
    summary = load_summary(data_dir, video_id)
    transcript = load_transcript(data_dir, video_id)
    record = video.to_dict()
    record.update(
        {
            "has_transcript": transcript is not None,
            "has_summary": summary is not None,
            "has_video": bool(video.video_path and Path(video.video_path).exists()),
            "has_audio": bool(video.audio_path and Path(video.audio_path).exists()),
            "has_subtitles": bool(
                video.subtitle_path and Path(video.subtitle_path).exists()
            ),
            "has_vectors": (vdir / VECTORS).exists(),
            "summary": summary.to_dict() if summary else None,
            "transcript_source": transcript.source if transcript else "",
            "segment_count": len(transcript.segments) if transcript else 0,
            "dir": str(vdir),
        }
    )
    return record


def list_videos(data_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(data_dir)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not (child / METADATA).exists():
            continue
        record = video_record(data_dir, child.name)
        if record is None:
            continue
        try:
            record["mtime"] = (child / METADATA).stat().st_mtime
        except OSError:
            record["mtime"] = 0
        records.append(record)
    records.sort(key=lambda r: r.get("mtime", 0), reverse=True)
    return records


def delete_video(data_dir: str | Path, video_id: str) -> bool:
    vdir = video_dir(data_dir, video_id)
    if not vdir.exists():
        return False
    shutil.rmtree(vdir)
    return True
