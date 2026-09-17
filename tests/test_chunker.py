from __future__ import annotations

from itertools import pairwise

from pipeline.chunker import (
    chunk_segments,
    chunk_text,
    chunks_from_transcript,
    clean_text,
    timestamp_range,
)
from pipeline.models import Chunk, Segment, Transcript


def test_clean_text_strips_vtt_tags_and_noise():
    raw = "<c>Hello</c>  \u200b world<00:00:01.000><c> again</c>\n\n\nNext"
    assert clean_text(raw) == "Hello world again\nNext"


def test_chunk_segments_respects_max_chars():
    segments = [Segment(i, i + 1, f"sentence {i}") for i in range(10)]
    chunks = chunk_segments(segments, max_chars=40, overlap_chars=0)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text.startswith("sentence 0")


def test_chunk_segments_cover_all_text(sample_segments):
    chunks = chunk_segments(sample_segments, max_chars=1000, overlap_chars=0)
    assert len(chunks) == 1
    joined = chunks[0].text
    for seg in sample_segments:
        assert seg.text.replace(".", "") in joined.replace(".", "")


def test_chunk_segments_timestamps(sample_segments):
    chunks = chunk_segments(sample_segments, max_chars=40, overlap_chars=0)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == sample_segments[-1].end
    for a, b in pairwise(chunks):
        assert a.end <= b.end


def test_overlap_creates_shared_text():
    segments = [Segment(i, i + 1, f"sentence {i}") for i in range(10)]
    with_overlap = chunk_segments(segments, max_chars=30, overlap_chars=15)
    assert len(with_overlap) >= 2
    first_words = set(with_overlap[0].text.split())
    second_words = set(with_overlap[1].text.split())
    assert first_words & second_words


def test_chunk_segments_empty():
    assert chunk_segments([]) == []
    assert chunk_segments([Segment(0, 1, "   ")]) == []


def test_chunk_text_splits_long_text():
    text = " ".join(f"Sentence number {i} about topic." for i in range(200))
    chunks = chunk_text(text, max_chars=300, overlap_chars=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.start == 0.0 and c.end == 0.0 for c in chunks)


def test_chunk_text_handles_one_huge_sentence():
    text = "word " * 500
    chunks = chunk_text(text, max_chars=100, overlap_chars=0)
    assert len(chunks) >= 5


def test_chunks_from_transcript_prefers_segments(sample_transcript):
    chunks = chunks_from_transcript(sample_transcript, max_chars=40, overlap_chars=0)
    assert chunks
    assert any(c.start or c.end for c in chunks)


def test_chunks_from_transcript_falls_back_to_text():
    t = Transcript(text="A. B. C. D.", segments=[], source="subs")
    chunks = chunks_from_transcript(t, max_chars=4, overlap_chars=0)
    assert chunks
    assert all(c.start == 0.0 and c.end == 0.0 for c in chunks)


def test_timestamp_range():
    assert timestamp_range(Chunk(0, 0.0, 0.0, "x")) is None
    assert timestamp_range(Chunk(0, 65.0, 130.0, "x")) == "01:05-02:10"
