"""Retrieval-augmented Q&A over transcript chunks with timestamp citations."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from pipeline.llm import LLMError, OllamaClient, strip_thinking
from pipeline.models import Answer, Chunk, Citation, Video, format_timestamp

ProgressFn = Callable[[str, float], None]

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

ANSWER_SYSTEM = (
    "You answer questions about a video using only the provided transcript "
    "excerpts. Cite timestamps in square brackets like [01:23] for every "
    "claim, using the markers given in the excerpts. If the excerpts do not "
    "cover the question, say so plainly. No disclaimers."
)


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
    prompt = (
        f"Video: {video.title} by {video.author}\n\n"
        f"Transcript excerpts:\n{format_context(citations)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above. Cite timestamps like [mm:ss]."
    )
    body = strip_thinking(
        llm.generate(prompt, model=model, system=ANSWER_SYSTEM, temperature=0.1)
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
