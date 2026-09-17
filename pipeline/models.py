"""Core data models shared by the pipeline, CLI, GUI, and agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Video:
    video_id: str
    url: str
    title: str
    author: str
    duration: float | None = None
    description: str = ""
    audio_path: str | None = None
    video_path: str | None = None
    subtitle_path: str | None = None
    thumbnail: str | None = None
    webpage_url: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Video:
        return cls(
            video_id=str(d.get("video_id") or d.get("id") or ""),
            url=d.get("url") or d.get("webpage_url") or "",
            title=d.get("title") or "Untitled",
            author=d.get("author") or d.get("uploader") or "Unknown",
            duration=_optional_float(d.get("duration")),
            description=d.get("description") or "",
            audio_path=d.get("audio_path"),
            video_path=d.get("video_path"),
            subtitle_path=d.get("subtitle_path"),
            thumbnail=d.get("thumbnail"),
            webpage_url=d.get("webpage_url") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    start: float
    end: float
    text: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Segment:
        return cls(start=_f(d.get("start")), end=_f(d.get("end")), text=d.get("text", ""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transcript:
    text: str = ""
    segments: list[Segment] = field(default_factory=list)
    source: str = ""
    language: str = ""
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "source": self.source,
            "language": self.language,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transcript:
        return cls(
            text=d.get("text", ""),
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            source=d.get("source", ""),
            language=d.get("language", ""),
            duration=_f(d.get("duration")),
        )


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(
            index=int(d.get("index", 0)),
            start=_f(d.get("start")),
            end=_f(d.get("end")),
            text=d.get("text", ""),
        )


@dataclass
class Summary:
    video_id: str = ""
    url: str = ""
    title: str = ""
    author: str = ""
    model: str = ""
    summary: str = ""
    highlights: list[str] = field(default_factory=list)
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Summary:
        return cls(
            video_id=d.get("video_id", ""),
            url=d.get("url", ""),
            title=d.get("title", ""),
            author=d.get("author", ""),
            model=d.get("model", ""),
            summary=d.get("summary", ""),
            highlights=list(d.get("highlights", [])),
            created=d.get("created", ""),
        )


@dataclass
class Citation:
    index: int
    start: float
    end: float
    text: str
    score: float = 0.0
    video_id: str = ""
    video_title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Citation:
        return cls(
            index=int(d.get("index", 0)),
            start=_f(d.get("start")),
            end=_f(d.get("end")),
            text=d.get("text", ""),
            score=_f(d.get("score")),
            video_id=d.get("video_id", "") or "",
            video_title=d.get("video_title", "") or "",
        )


@dataclass
class Answer:
    video_id: str = ""
    question: str = ""
    answer: str = ""
    model: str = ""
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "question": self.question,
            "answer": self.answer,
            "model": self.model,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class ChatMessage:
    role: str = "user"
    content: str = ""
    mode: str = "ask"
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    created: str = ""
    prompt_tokens: int = 0
    eval_tokens: int = 0
    context_limit: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "mode": self.mode,
            "citations": [c.to_dict() for c in self.citations],
            "model": self.model,
            "created": self.created,
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
            "context_limit": self.context_limit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatMessage:
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            mode=d.get("mode", "ask"),
            citations=[Citation.from_dict(c) for c in d.get("citations", [])],
            model=d.get("model", ""),
            created=d.get("created", ""),
            prompt_tokens=int(d.get("prompt_tokens") or 0),
            eval_tokens=int(d.get("eval_tokens") or 0),
            context_limit=int(d.get("context_limit") or 0),
        )


@dataclass
class ChatThread:
    thread_id: str = ""
    scope: str = "video"
    video_id: str = ""
    title: str = ""
    mode: str = "ask"
    playlist_ids: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    messages: list[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "scope": self.scope,
            "video_id": self.video_id,
            "title": self.title,
            "mode": self.mode,
            "playlist_ids": list(self.playlist_ids),
            "created": self.created,
            "updated": self.updated,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatThread:
        return cls(
            thread_id=d.get("thread_id", ""),
            scope=d.get("scope", "video"),
            video_id=d.get("video_id", "") or "",
            title=d.get("title", ""),
            mode=d.get("mode", "ask"),
            playlist_ids=list(d.get("playlist_ids") or []),
            created=d.get("created", ""),
            updated=d.get("updated", ""),
            messages=[ChatMessage.from_dict(m) for m in d.get("messages", [])],
        )


@dataclass
class Playlist:
    playlist_id: str = ""
    name: str = ""
    video_ids: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "playlist_id": self.playlist_id,
            "name": self.name,
            "video_ids": list(self.video_ids),
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Playlist:
        return cls(
            playlist_id=d.get("playlist_id", ""),
            name=d.get("name", ""),
            video_ids=list(d.get("video_ids") or []),
            created=d.get("created", ""),
            updated=d.get("updated", ""),
        )


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for progress displays."""
    return max(1, len(text or "") // 4)


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
