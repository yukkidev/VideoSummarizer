"""Per-video artifact persistence under data/<video_id>/."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.models import (
    Answer,
    ChatMessage,
    ChatThread,
    Chunk,
    Citation,
    Playlist,
    Summary,
    Transcript,
    Video,
)

METADATA = "metadata.json"
TRANSCRIPT = "transcript.json"
CHUNKS = "chunks.json"
SUMMARY = "summary.json"
SUMMARY_MD = "summary.md"
VECTORS = "vectors.json"
ANSWERS = "answers.jsonl"
CHATS = "chats"
PLAYLISTS = "playlists"


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def chats_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / CHATS


def thread_path(data_dir: str | Path, thread_id: str) -> Path:
    return chats_dir(data_dir) / f"{thread_id}.json"


def save_thread(data_dir: str | Path, thread: ChatThread) -> Path:
    path = thread_path(data_dir, thread.thread_id)
    _write_json(path, thread.to_dict())
    return path


def load_thread(data_dir: str | Path, thread_id: str) -> ChatThread | None:
    raw = _read_json(thread_path(data_dir, thread_id))
    return ChatThread.from_dict(raw) if isinstance(raw, dict) else None


def thread_summary(thread: ChatThread) -> dict[str, Any]:
    return {
        "thread_id": thread.thread_id,
        "scope": thread.scope,
        "video_id": thread.video_id,
        "title": thread.title,
        "mode": thread.mode,
        "created": thread.created,
        "updated": thread.updated,
        "message_count": len(thread.messages),
    }


def list_threads(
    data_dir: str | Path,
    *,
    video_id: str | None = None,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    root = chats_dir(data_dir)
    if not root.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_file() or child.suffix != ".json":
            continue
        raw = _read_json(child)
        if not isinstance(raw, dict):
            continue
        thread = ChatThread.from_dict(raw)
        if scope is not None and thread.scope != scope:
            continue
        if video_id is not None and thread.video_id != video_id:
            continue
        summaries.append(thread_summary(thread))
    summaries.sort(key=lambda t: t.get("updated", ""), reverse=True)
    return summaries


def delete_thread(data_dir: str | Path, thread_id: str) -> bool:
    path = thread_path(data_dir, thread_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def delete_threads_for_video(data_dir: str | Path, video_id: str) -> int:
    deleted = 0
    for summary in list_threads(data_dir, video_id=video_id):
        if delete_thread(data_dir, summary["thread_id"]):
            deleted += 1
    return deleted


def playlists_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / PLAYLISTS


def playlist_path(data_dir: str | Path, playlist_id: str) -> Path:
    return playlists_dir(data_dir) / f"{playlist_id}.json"


def save_playlist(data_dir: str | Path, playlist: Playlist) -> Path:
    path = playlist_path(data_dir, playlist.playlist_id)
    _write_json(path, playlist.to_dict())
    return path


def load_playlist(data_dir: str | Path, playlist_id: str) -> Playlist | None:
    raw = _read_json(playlist_path(data_dir, playlist_id))
    return Playlist.from_dict(raw) if isinstance(raw, dict) else None


def list_playlists(data_dir: str | Path) -> list[Playlist]:
    root = playlists_dir(data_dir)
    if not root.exists():
        return []
    playlists: list[Playlist] = []
    for child in root.iterdir():
        if not child.is_file() or child.suffix != ".json":
            continue
        raw = _read_json(child)
        if isinstance(raw, dict):
            playlists.append(Playlist.from_dict(raw))
    playlists.sort(key=lambda p: p.name.lower())
    return playlists


def delete_playlist(data_dir: str | Path, playlist_id: str) -> bool:
    path = playlist_path(data_dir, playlist_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def remove_video_from_playlists(data_dir: str | Path, video_id: str) -> int:
    touched = 0
    for playlist in list_playlists(data_dir):
        if video_id in playlist.video_ids:
            playlist.video_ids = [v for v in playlist.video_ids if v != video_id]
            playlist.updated = _now()
            save_playlist(data_dir, playlist)
            touched += 1
    return touched


def playlist_video_ids(data_dir: str | Path, playlist_ids: list[str]) -> list[str]:
    """Union of video ids across the given playlists, de-duplicated in order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for playlist_id in playlist_ids:
        playlist = load_playlist(data_dir, playlist_id)
        if playlist is None:
            continue
        for video_id in playlist.video_ids:
            if video_id not in seen:
                seen.add(video_id)
                ordered.append(video_id)
    return ordered


def migrate_answers(data_dir: str | Path, video_id: str) -> str | None:
    """Convert a legacy answers.jsonl into one chat thread, once."""
    legacy = video_dir(data_dir, video_id) / ANSWERS
    if not legacy.exists():
        return None
    records = load_answers(data_dir, video_id)
    first_at = next((r.get("created") or "" for r in records if r.get("created")), "")
    last_at = next((r.get("created") or "" for r in reversed(records) if r.get("created")), "")
    thread = ChatThread(
        thread_id=uuid.uuid4().hex[:12],
        scope="video",
        video_id=video_id,
        title="Earlier questions",
        mode="ask",
        created=first_at or _now(),
        updated=last_at or _now(),
    )
    for record in records:
        question = str(record.get("question") or "").strip()
        answer = str(record.get("answer") or "").strip()
        created = record.get("created") or ""
        citations = [
            Citation.from_dict(c)
            for c in record.get("citations") or []
            if isinstance(c, dict)
        ]
        if question:
            thread.messages.append(
                ChatMessage(role="user", content=question, mode="ask", created=created)
            )
        if answer:
            thread.messages.append(
                ChatMessage(
                    role="assistant",
                    content=answer,
                    mode="ask",
                    citations=citations,
                    model=record.get("model") or "",
                    created=created,
                )
            )
    if not thread.messages:
        return None
    save_thread(data_dir, thread)
    try:
        legacy.rename(legacy.with_suffix(".jsonl.migrated"))
    except OSError:
        pass
    return thread.thread_id


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
