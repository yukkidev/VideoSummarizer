from __future__ import annotations

import json

import pytest

from pipeline.llm import LLMError, OllamaClient
from pipeline.models import Chunk, Video
from pipeline.summarizer import (
    _batch_chunks,
    _parse_final,
    _split_sections,
    summarize_chunks,
    summary_to_markdown,
)


@pytest.fixture()
def video() -> Video:
    return Video(
        video_id="v1",
        url="https://example.com/v1",
        title="Test Video",
        author="Tester",
        duration=600,
    )


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of responses")
        return self.responses.pop(0)


def client_for(fake: FakeLLM) -> OllamaClient:
    return OllamaClient(generate=fake.generate, model="test-model")


def make_chunks(n: int, size: int = 100) -> list[Chunk]:
    return [
        Chunk(index=i, start=i * 10.0, end=(i + 1) * 10.0, text="word " * (size // 5))
        for i in range(n)
    ]


def test_parse_final_json():
    body = '{"summary": "- a\\n- b", "questions": ["q1?", "q2?"]}'
    summary, questions = _parse_final(body)
    assert "- a" in summary
    assert questions == ["q1?", "q2?"]


OPEN = "<" + "think" + ">"
CLOSE = "<" + "/" + "think" + ">"


def test_parse_final_json_in_fences_with_thinking():
    body = f"{OPEN}reasoning{CLOSE}\n```json\n{{\"summary\": \"s\", \"questions\": [\"q\"]}}\n```"
    assert _parse_final(body) == ("s", ["q"])


def test_parse_final_summary_as_list():
    body = json.dumps({"summary": ["one", "two"], "questions": "q1\nq2"})
    summary, questions = _parse_final(body)
    assert "• one" in summary
    assert questions == ["q1", "q2"]


def test_parse_final_fallback_sections():
    body = (
        "SUMMARY:\n"
        "- first point\n"
        "- second point\n\n"
        "QUESTIONS:\n"
        "1. why?\n"
        "2. how?\n"
    )
    summary, questions = _parse_final(body)
    assert "first point" in summary
    assert questions == ["why?", "how?"]


def test_split_sections_ignores_question_markers():
    summary, questions = _split_sections("TOP QUESTIONS\n- a?\n- b?")
    assert summary == ""
    assert questions == ["a?", "b?"]


def test_batch_chunks_groups_by_size():
    chunks = make_chunks(10, size=100)
    batches = _batch_chunks(chunks, batch_chars=250)
    assert len(batches) > 1
    assert sum(len(b) for b in batches) == 10
    assert [c.index for b in batches for c in b] == list(range(10))


def test_single_pass_small_transcript(video):
    fake = FakeLLM(['{"summary": "the summary", "questions": ["q1"]}'])
    llm = client_for(fake)
    chunks = [Chunk(index=0, start=0, end=30, text="short transcript")]
    result = summarize_chunks(chunks, video, llm)
    assert result.summary == "the summary"
    assert result.highlights == ["q1"]
    assert result.model == "test-model"
    assert result.video_id == "v1"
    assert result.created
    assert len(fake.calls) == 1
    assert fake.calls[0]["json_mode"] is True


def test_map_reduce_for_long_transcript(video):
    fake = FakeLLM([
        "notes one",
        "notes two",
        "notes three",
        '{"summary": "final", "questions": ["q"]}',
    ])
    llm = client_for(fake)
    chunks = make_chunks(12, size=200)
    result = summarize_chunks(chunks, video, llm, batch_chars=1000)
    map_calls = [c for c in fake.calls if not c.get("json_mode")]
    reduce_calls = [c for c in fake.calls if c.get("json_mode")]
    assert len(map_calls) >= 2
    assert len(reduce_calls) == 1
    assert result.summary == "final"
    assert "[00:00-00:10]" in map_calls[0]["prompt"]


def test_map_reduce_prompt_contains_video_metadata(video):
    fake = FakeLLM(["notes", "notes two", '{"summary": "s", "questions": []}'])
    llm = client_for(fake)
    chunks = make_chunks(3, size=400)
    summarize_chunks(chunks, video, llm, batch_chars=1000)
    final_prompt = fake.calls[-1]["prompt"]
    assert "Test Video" in final_prompt
    assert "Tester" in final_prompt


def test_empty_chunks(video):
    fake = FakeLLM([])
    llm = client_for(fake)
    result = summarize_chunks([], video, llm)
    assert "No transcript content" in result.summary
    assert fake.calls == []


def test_llm_error_is_captured(video):
    class Boom(FakeLLM):
        def generate(self, prompt, **kwargs):
            raise LLMError("ollama exploded")

    llm = client_for(Boom([]))
    chunks = [Chunk(index=0, start=0, end=5, text="hi")]
    result = summarize_chunks(chunks, video, llm)
    assert "Summarization failed" in result.summary
    assert "ollama exploded" in result.summary


def test_progress_callback(video):
    fake = FakeLLM(["notes", "notes two", '{"summary": "s", "questions": []}'])
    llm = client_for(fake)
    seen: list[tuple[str, float]] = []
    chunks = make_chunks(6, size=200)
    summarize_chunks(chunks, video, llm, batch_chars=1000, progress=lambda m, p: seen.append((m, p)))
    assert seen
    assert seen[-1] == ("summarizing", 1.0)


def test_summary_to_markdown(video):
    fake = FakeLLM(['{"summary": "- point", "questions": ["why?"]}'])
    result = summarize_chunks([Chunk(0, 0, 5, "text")], video, client_for(fake))
    md = summary_to_markdown(
        result,
        [Chunk(0, 5.0, 15.0, "chunk text")],
    )
    assert "# Test Video" in md
    assert "- point" in md
    assert "why?" in md
    assert "00:05-00:15" in md
