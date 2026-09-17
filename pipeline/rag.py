"""Retrieval-augmented Q&A over transcript chunks with timestamp citations.

Two reply modes share the same retrieval:
- ``ask``  — strictly grounded in the retrieved excerpts, timestamp-cited
- ``chat`` — conversational, history-aware, may build beyond the excerpts
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pipeline.llm import LLMError, OllamaClient, strip_thinking
from pipeline.models import (
    Answer,
    ChatMessage,
    Chunk,
    Citation,
    Video,
    format_timestamp,
)

ProgressFn = Callable[[str, float], None]

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

ANSWER_SYSTEM = (
    "You answer questions about a video using only the provided transcript "
    "excerpts. Cite timestamps in square brackets like [01:23] for every "
    "claim, using the markers given in the excerpts. If the excerpts do not "
    "cover the question, say so plainly. No disclaimers."
)

CHAT_SYSTEM = (
    "You are a thoughtful learning companion for the user's personal video "
    "library. You are having an ongoing conversation about what the user is "
    "learning, and you remember the earlier turns. When transcript excerpts "
    "are provided, use them and cite timestamps like [01:23] for anything "
    "drawn from the videos; you may also reason with the user, connect ideas "
    "across the conversation, and use your own general knowledge. Be "
    "conversational and concise. No disclaimers."
)

HISTORY_MAX_MESSAGES = 12
HISTORY_MAX_CHARS = 9000


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def build_vectors(
    chunks: Sequence[Chunk],
    llm: OllamaClient,
    *,
    embed_model: str | None = None,
    batch_size: int = 16,
    progress: ProgressFn | None = None,
) -> list[list[float]]:
    notify = progress or (lambda _m, _p: None)
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [DOC_PREFIX + c.text for c in batch]
        notify(
            f"embedding chunks {start + 1}-{start + len(batch)}/{len(chunks)}",
            (start / len(chunks)) if chunks else 1.0,
        )
        vectors.extend(llm.embed(texts, model=embed_model))
    notify("embedding chunks", 1.0)
    if len(vectors) != len(chunks):
        raise LLMError(
            f"embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
        )
    return vectors


def search(
    question: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    llm: OllamaClient,
    *,
    top_k: int = 6,
    embed_model: str | None = None,
) -> list[Citation]:
    if not chunks or not vectors:
        return []
    query_vector = llm.embed([QUERY_PREFIX + question], model=embed_model)[0]
    scored: list[Citation] = []
    for chunk, vector in zip(chunks, vectors):
        scored.append(
            Citation(
                index=chunk.index,
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
                score=round(cosine(query_vector, vector), 6),
            )
        )
    scored.sort(key=lambda c: (-c.score, c.start))
    return scored[: max(1, top_k)]


def format_context(citations: Sequence[Citation]) -> str:
    blocks = []
    for c in citations:
        stamp = f"{format_timestamp(c.start)}-{format_timestamp(c.end)}"
        blocks.append(f"[{stamp}] {c.text}")
    return "\n\n".join(blocks)


def format_global_context(citations: Sequence[Citation]) -> str:
    blocks = []
    for c in citations:
        stamp = f"{format_timestamp(c.start)}-{format_timestamp(c.end)}"
        label = f"{c.video_title} @ {stamp}" if c.video_title else stamp
        blocks.append(f"[{label}] {c.text}")
    return "\n\n".join(blocks)


def trim_history(
    messages: Sequence[ChatMessage],
    *,
    max_messages: int = HISTORY_MAX_MESSAGES,
    max_chars: int = HISTORY_MAX_CHARS,
) -> list[dict[str, str]]:
    """Tail of the conversation as chat messages, trimmed to fit context."""
    selected: list[dict[str, str]] = []
    used = 0
    for message in reversed(list(messages)):
        role = getattr(message, "role", "")
        content = (getattr(message, "content", "") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if len(selected) >= max_messages:
            break
        if selected and used + len(content) > max_chars:
            break
        selected.append({"role": role, "content": content})
        used += len(content)
    selected.reverse()
    return selected


def build_reply_prompt(
    question: str,
    citations: Sequence[Citation],
    *,
    mode: str = "ask",
    video: Video | None = None,
    global_scope: bool = False,
    scope_label: str | None = None,
) -> str:
    context = (
        format_global_context(citations) if global_scope else format_context(citations)
    )
    if global_scope:
        head = f"Scope: {scope_label or 'the whole video library'}\n\n"
    elif video is not None:
        head = f"Video: {video.title} by {video.author}\n\n"
    else:
        head = ""
    if mode == "ask":
        return (
            f"{head}Transcript excerpts:\n{context or '(no relevant excerpts found)'}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the excerpts above. Cite timestamps like [mm:ss]."
        )
    return (
        f"{head}Relevant transcript excerpts (use when helpful):\n"
        f"{context or '(none retrieved)'}\n\n"
        f"User: {question}\n\n"
        "Continue the conversation. Cite timestamps like [mm:ss] when you "
        "reference the videos."
    )


def generate_reply(
    question: str,
    *,
    mode: str,
    history: Sequence[ChatMessage],
    citations: Sequence[Citation],
    llm: OllamaClient,
    video: Video | None = None,
    global_scope: bool = False,
    scope_label: str | None = None,
    model: str | None = None,
) -> str:
    content, _usage = generate_reply_detailed(
        question,
        mode=mode,
        history=history,
        citations=citations,
        llm=llm,
        video=video,
        global_scope=global_scope,
        scope_label=scope_label,
        model=model,
    )
    return content


def generate_reply_detailed(
    question: str,
    *,
    mode: str,
    history: Sequence[ChatMessage],
    citations: Sequence[Citation],
    llm: OllamaClient,
    video: Video | None = None,
    global_scope: bool = False,
    scope_label: str | None = None,
    model: str | None = None,
) -> tuple[str, dict[str, int]]:
    """Generate a reply and report Ollama's token usage when available."""
    system = CHAT_SYSTEM if mode == "chat" else ANSWER_SYSTEM
    messages = [{"role": "system", "content": system}]
    messages.extend(trim_history(history))
    messages.append(
        {
            "role": "user",
            "content": build_reply_prompt(
                question, citations, mode=mode, video=video,
                global_scope=global_scope, scope_label=scope_label,
            ),
        }
    )
    temperature = 0.4 if mode == "chat" else 0.1
    detailed = getattr(llm, "chat_detailed", None)
    if callable(detailed):
        result = detailed(messages, model=model, temperature=temperature)
        return str(result.get("content") or ""), {
            "prompt_tokens": int(result.get("prompt_tokens") or 0),
            "eval_tokens": int(result.get("eval_tokens") or 0),
        }
    content = strip_thinking(
        llm.chat(messages, model=model, temperature=temperature)
    )
    return content, {"prompt_tokens": 0, "eval_tokens": 0}


def search_all(
    question: str,
    *,
    data_dir: str | Path,
    llm: OllamaClient,
    top_k: int = 6,
    embed_model: str | None = None,
    video_ids: Sequence[str] | None = None,
) -> list[Citation]:
    """Rank chunks across processed videos by cosine similarity.

    ``video_ids`` restricts the search to those videos (e.g. a playlist scope);
    ``None`` searches the whole library.
    """
    from pipeline import store  # local import: store must not depend on rag

    question = (question or "").strip()
    if not question:
        return []
    query_vector = llm.embed([QUERY_PREFIX + question], model=embed_model)[0]
    wanted_model = embed_model or llm.embed_model
    wanted_ids = set(video_ids) if video_ids is not None else None
    root = Path(data_dir)
    if not root.exists():
        return []
    scored: list[Citation] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / store.METADATA).exists():
            continue
        if wanted_ids is not None and child.name not in wanted_ids:
            continue
        video_id = child.name
        chunks = store.load_chunks(data_dir, video_id) or []
        stored = store.load_vectors(data_dir, video_id) or {}
        vectors = stored.get("vectors") or []
        if not chunks or len(vectors) != len(chunks):
            continue
        if stored.get("model") != wanted_model:
            continue
        video = store.load_video(data_dir, video_id)
        title = (video.title if video else "") or video_id
        for chunk, vector in zip(chunks, vectors):
            scored.append(
                Citation(
                    index=chunk.index,
                    start=chunk.start,
                    end=chunk.end,
                    text=chunk.text,
                    score=round(cosine(query_vector, vector), 6),
                    video_id=video_id,
                    video_title=title,
                )
            )
    scored.sort(key=lambda c: (-c.score, c.video_id, c.start))
    return scored[: max(1, top_k)]


def answer_question(
    video: Video,
    chunks: Sequence[Chunk],
    question: str,
    llm: OllamaClient,
    *,
    vectors: Sequence[Sequence[float]] | None = None,
    top_k: int = 6,
    model: str | None = None,
    embed_model: str | None = None,
    progress: ProgressFn | None = None,
) -> Answer:
    notify = progress or (lambda _m, _p: None)
    if not chunks:
        return Answer(
            video_id=video.video_id,
            question=question,
            answer="This video has no transcript to search.",
            model=model or llm.model,
            citations=[],
        )
    notify("retrieving relevant moments", 0.1)
    if vectors is None:
        vectors = build_vectors(chunks, llm, embed_model=embed_model, progress=progress)
    citations = search(
        question, chunks, vectors, llm, top_k=top_k, embed_model=embed_model,
    )
    notify("writing answer", 0.5)
    body = generate_reply(
        question,
        mode="ask",
        history=(),
        citations=citations,
        llm=llm,
        video=video,
        model=model,
    )
    notify("writing answer", 1.0)
    return Answer(
        video_id=video.video_id,
        question=question,
        answer=body or "The model returned an empty answer.",
        model=model or llm.model,
        citations=citations,
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
