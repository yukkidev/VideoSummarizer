from __future__ import annotations

import pytest

from pipeline.llm import OllamaClient
from pipeline.models import Chunk, Video
from pipeline.rag import (
    QUERY_PREFIX,
    answer_question,
    build_vectors,
    cosine,
    format_context,
    search,
)


class FakeEmbedLLM:
    model = "fake-model"

    def __init__(self, table, vectors=None):
        self.table = table
        self.vectors = vectors
        self.embed_calls = []
        self.generate_calls = []

    def embed(self, texts, model=None, timeout=None):
        self.embed_calls.append((list(texts), model))
        if self.vectors is not None:
            return [list(v) for v in self.vectors]
        out = []
        for text in texts:
            key = text.replace("search_document: ", "").replace("search_query: ", "")
            out.append(self.table.get(key, [0.0, 0.0, 0.0]))
        return out

    def generate(self, prompt, **kwargs):
        self.generate_calls.append((prompt, kwargs))
        return "Because of X [00:05]. See also Y [00:15]."


def chunks_for(texts):
    return [Chunk(index=i, start=i * 10.0, end=i * 10.0 + 10, text=t) for i, t in enumerate(texts)]


def test_cosine_identical():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_and_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_build_vectors_prefixes_documents():
    llm = FakeEmbedLLM({"a": [1, 0, 0], "b": [0, 1, 0]})
    vectors = build_vectors(chunks_for(["a", "b"]), llm, batch_size=1)
    assert vectors == [[1, 0, 0], [0, 1, 0]]
    assert all(call[0][0].startswith("search_document:") for call in llm.embed_calls)
    assert len(llm.embed_calls) == 2


def test_build_vectors_count_mismatch():
    llm = FakeEmbedLLM({}, vectors=[[1.0]])
    with pytest.raises(Exception, match="mismatch"):
        build_vectors(chunks_for(["a", "b"]), llm)


def test_search_ranks_by_similarity():
    table = {
        "cats": [1.0, 0.0, 0.0],
        "dogs": [0.0, 1.0, 0.0],
        "birds": [0.0, 0.0, 1.0],
    }
    llm = FakeEmbedLLM(table)
    chunks = chunks_for(["cats", "dogs", "birds"])
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    results = search("cats", chunks, vectors, llm, top_k=2)
    assert results[0].text == "cats"
    assert results[0].score > results[1].score
    assert len(results) == 2
    assert llm.embed_calls[0][0][0] == QUERY_PREFIX + "cats"


def test_search_empty_inputs():
    llm = FakeEmbedLLM({})
    assert search("q", [], [], llm) == []


def test_format_context_includes_timestamps():
    citations = [
        type("C", (), {"start": 65.0, "end": 130.0, "text": "hello"})(),
    ]
    assert format_context(citations) == "[01:05-02:10] hello"


def test_answer_question_with_fake_llm():
    llm = FakeEmbedLLM({"cats": [1.0, 0.0, 0.0], "question": [1.0, 0.0, 0.0]})
    chunks = chunks_for(["cats are great", "unrelated"])
    video = Video(video_id="v1", url="u", title="T", author="A")
    answer = answer_question(video, chunks, "question", llm, top_k=1)
    assert answer.video_id == "v1"
    assert "[00:05]" in answer.answer
    assert len(answer.citations) == 1
    assert "cats are great" in llm.generate_calls[0][0]


def test_answer_question_no_chunks():
    llm = FakeEmbedLLM({})
    video = Video(video_id="v1", url="u", title="T", author="A")
    answer = answer_question(video, [], "q", llm)
    assert "no transcript" in answer.answer
    assert answer.citations == []


def test_answer_question_builds_vectors_when_missing():
    llm = FakeEmbedLLM({"a": [1.0, 0.0, 0.0], "q": [1.0, 0.0, 0.0]})
    chunks = chunks_for(["a"])
    video = Video(video_id="v1", url="u", title="T", author="A")
    answer = answer_question(video, chunks, "q", llm)
    assert answer.answer
    assert len(llm.embed_calls) == 2


def test_ollama_client_generate_override_matches_fake_interface():
    client = OllamaClient(generate=lambda prompt, **kw: "ok")
    assert client.generate("hi") == "ok"
