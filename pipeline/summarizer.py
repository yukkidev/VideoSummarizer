"""Map-reduce summarization over transcript chunks via a local LLM."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from pipeline.chunker import timestamp_range
from pipeline.llm import LLMError, OllamaClient, strip_thinking
from pipeline.models import Chunk, Summary, Video

ProgressFn = Callable[[str, float], None]

DEFAULT_BATCH_CHARS = 6000
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

MAP_SYSTEM = (
    "You are a meticulous video analyst. You extract concrete, specific key "
    "points. No filler, no moralizing, no disclaimers."
)
REDUCE_SYSTEM = (
    "You are a meticulous video analyst writing for a curious viewer. "
    "Be specific to the video. No moralizing, no disclaimers."
)


def _format_chunk(chunk: Chunk) -> str:
    stamp = timestamp_range(chunk)
    prefix = f"[{stamp}] " if stamp else ""
    return f"{prefix}{chunk.text}"


def _parse_final(body: str) -> tuple[str, list[str]]:
    body = strip_thinking(body)
    candidate = body
    fence = JSON_FENCE_RE.search(body)
    if fence:
        candidate = fence.group(1)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            summary = data.get("summary") or ""
            if isinstance(summary, list):
                summary = "\n".join(f"• {s}" for s in summary)
            questions = data.get("questions") or data.get("highlights") or []
            if isinstance(questions, str):
                questions = [
                    q.strip(" -*•\t") for q in questions.splitlines() if q.strip()
                ]
            cleaned = [str(q).strip() for q in questions if str(q).strip()]
            if summary or cleaned:
                return str(summary).strip(), cleaned
    return _split_sections(body)


def _split_sections(body: str) -> tuple[str, list[str]]:
    summary_lines: list[str] = []
    questions: list[str] = []
    in_questions = False
    for line in body.splitlines():
        stripped = line.strip()
        if re.match(r"^\s*(#{1,4}\s*)?(summary|key points|takeaways)\b", stripped, re.IGNORECASE):
            in_questions = False
            rest = re.sub(
                r"^\s*(#{1,4}\s*)?(summary|key points|takeaways)[:\s-]*", "", stripped, flags=re.IGNORECASE
            )
            if rest:
                summary_lines.append(rest)
            continue
        if re.match(r"^\s*(#{1,4}\s*)?(top questions|questions|q&a)\b", stripped, re.IGNORECASE):
            in_questions = True
            continue
        if not stripped:
            continue
        if in_questions:
            questions.append(stripped.lstrip("-*•0123456789. \t"))
        else:
            summary_lines.append(stripped)
    return "\n".join(summary_lines).strip(), [q for q in questions if q]


def _batch_chunks(
    chunks: Sequence[Chunk], batch_chars: int,
) -> list[list[Chunk]]:
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_len = 0
    for chunk in chunks:
        length = len(chunk.text) + 20
        if current and current_len + length > batch_chars:
            batches.append(current)
            current, current_len = [], 0
        current.append(chunk)
        current_len += length
    if current:
        batches.append(current)
    return batches


def _map_batch(batch: Sequence[Chunk], llm: OllamaClient, model: str | None) -> str:
    section = "\n\n".join(_format_chunk(c) for c in batch)
    prompt = (
        "Extract the key points from this section of a video transcript. "
        "Write 2-5 concise bullets. Keep names, numbers, and any timestamps "
        "shown in [mm:ss-mm:ss] markers. Do not add commentary.\n\n"
        f"TRANSCRIPT SECTION:\n{section}\n\nKEY POINTS:"
    )
    return llm.generate(prompt, model=model, system=MAP_SYSTEM, temperature=0.1)


def _reduce_notes(
    notes: Sequence[str],
    video: Video,
    llm: OllamaClient,
    model: str | None,
) -> tuple[str, list[str]]:
    combined = "\n\n".join(f"Section {i + 1} notes:\n{n}" for i, n in enumerate(notes))
    prompt = (
        "Below are analyst notes from consecutive sections of one video. "
        "Produce a final result as strict JSON with exactly these keys:\n"
        '{"summary": "3-6 bullet points separated by newlines, each a real '
        'takeaway with detail", "questions": ["5-8 insightful questions a '
        'curious viewer would want answered"]}\n\n'
        "Rules: do not answer the questions, do not invent facts, be specific "
        "to THIS video.\n\n"
        f"VIDEO TITLE: {video.title}\n"
        f"AUTHOR: {video.author}\n\n"
        f"NOTES:\n{combined}\n\nJSON:"
    )
    body = llm.generate(
        prompt, model=model, system=REDUCE_SYSTEM, temperature=0.2, json_mode=True,
    )
    return _parse_final(body)


def _single_pass(
    chunks: Sequence[Chunk],
    video: Video,
    llm: OllamaClient,
    model: str | None,
) -> tuple[str, list[str]]:
    section = "\n\n".join(_format_chunk(c) for c in chunks)
    prompt = (
        "Analyze this video transcript and return strict JSON with exactly "
        'these keys: {"summary": "3-6 bullet points separated by newlines", '
        '"questions": ["5-8 insightful questions"]}. Do not answer the '
        "questions. Be specific to THIS video.\n\n"
        f"VIDEO TITLE: {video.title}\n"
        f"AUTHOR: {video.author}\n\n"
        f"TRANSCRIPT:\n{section}\n\nJSON:"
    )
    body = llm.generate(
        prompt, model=model, system=REDUCE_SYSTEM, temperature=0.2, json_mode=True,
    )
    return _parse_final(body)


def summarize_chunks(
    chunks: Sequence[Chunk],
    video: Video,
    llm: OllamaClient,
    *,
    model: str | None = None,
    batch_chars: int = DEFAULT_BATCH_CHARS,
    progress: ProgressFn | None = None,
) -> Summary:
    notify = progress or (lambda _m, _p: None)
    model_name = model or llm.model

    if not chunks:
        return Summary(
            video_id=video.video_id,
            url=video.url,
            title=video.title,
            author=video.author,
            model=model_name,
            summary="No transcript content was available to summarize.",
            highlights=[],
            created=_now(),
        )

    total_chars = sum(len(c.text) for c in chunks)
    notify("summarizing", 0.05)
    try:
        if total_chars <= batch_chars and len(chunks) <= 8:
            summary_text, questions = _single_pass(chunks, video, llm, model)
        else:
            batches = _batch_chunks(chunks, batch_chars)
            notes: list[str] = []
            for i, batch in enumerate(batches):
                notify(f"summarizing section {i + 1}/{len(batches)}", 0.05 + 0.85 * (i / len(batches)))
                notes.append(_map_batch(batch, llm, model))
            summary_text, questions = _reduce_notes(notes, video, llm, model)
    except LLMError as exc:
        return Summary(
            video_id=video.video_id,
            url=video.url,
            title=video.title,
            author=video.author,
            model=model_name,
            summary=f"Summarization failed: {exc}",
            highlights=[],
            created=_now(),
        )

    notify("summarizing", 1.0)
    return Summary(
        video_id=video.video_id,
        url=video.url,
        title=video.title,
        author=video.author,
        model=model_name,
        summary=summary_text or "The model returned an empty summary.",
        highlights=questions,
        created=_now(),
    )


def summary_to_markdown(summary: Summary, chunks: Sequence[Chunk] = ()) -> str:
    lines = [
        f"# {summary.title}",
        "",
        f"- Author: {summary.author}",
        f"- Source: {summary.url}",
        f"- Model: {summary.model}",
        f"- Generated: {summary.created}",
        "",
        "## Summary",
        "",
        summary.summary,
        "",
        "## Questions to Explore",
        "",
    ]
    lines.extend(f"- {q}" for q in summary.highlights)
    if chunks:
        lines.extend(["", "## Timestamped Transcript", ""])
        for chunk in chunks:
            stamp = timestamp_range(chunk)
            if stamp:
                lines.append(f"### {stamp}")
            else:
                lines.append("### Section")
            lines.extend(["", chunk.text, ""])
    return "\n".join(lines).rstrip() + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
