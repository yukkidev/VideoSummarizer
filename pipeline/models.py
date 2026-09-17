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


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
