from __future__ import annotations

import io
import json
import subprocess
import urllib.error

import pytest

import pipeline.llm as llm_mod
from pipeline.llm import (
    LLMError,
    OllamaClient,
    strip_thinking,
)
from tests.conftest import completed


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def http(monkeypatch):
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode()) if req.data else None
        calls.append({"url": req.full_url, "body": body, "method": req.method})
        if req.full_url.endswith("/api/tags"):
            return FakeResponse({
                "models": [
                    {"name": "ornith-1.5:9b", "size": 6_600_000_000,
                     "modified_at": "2026-01-01", "digest": "abc"},
                    {"name": "nomic-embed-text:latest", "size": 274_000_000},
                ]
            })
        if req.full_url.endswith("/api/ps"):
            return FakeResponse({
                "models": [{"name": "ornith-1.5:9b", "size_vram": 7_100_000_000,
                            "expires_at": "2026-01-01T00:05:00Z",
                            "context_length": 100000}]
            })
        if req.full_url.endswith("/api/version"):
            return FakeResponse({"version": "0.33.3"})
        if req.full_url.endswith("/api/show"):
            return FakeResponse({
                "model_info": {
                    "general.architecture": "llama",
                    "llama.context_length": 262144,
                },
                "parameters": "temperature 1",
            })
        if req.full_url.endswith("/api/generate"):
            return FakeResponse({"response": "hello there"})
        if req.full_url.endswith("/api/chat"):
            return FakeResponse({
                "model": "ornith-1.5:9b",
                "message": {"content": "chat reply"},
                "prompt_eval_count": 17,
                "eval_count": 5,
                "done_reason": "stop",
            })
        if req.full_url.endswith("/api/embed"):
            inputs = body.get("input", [])
            return FakeResponse({"embeddings": [[0.1, 0.2] for _ in inputs]})
        raise AssertionError(f"unexpected url {req.full_url}")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    return calls


OPEN = "<" + "think" + ">"
CLOSE = "<" + "/" + "think" + ">"


def test_strip_thinking():
    assert strip_thinking(f"{OPEN}reasoning{CLOSE}final answer") == "final answer"
    assert strip_thinking("plain") == "plain"
    assert strip_thinking(f"reasoning here\n{CLOSE}\n\n4") == "4"
    assert strip_thinking(f"{OPEN}only thinking{CLOSE}") == ""
    assert strip_thinking(f"{OPEN}a{CLOSE}middle{OPEN}b{CLOSE}") == "middle"
    assert strip_thinking(f"{OPEN}cut off mid-thought") == ""


def test_health(http):
    assert OllamaClient().health() is True


def test_health_false_when_down(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", boom)
    assert OllamaClient().health() is False


def test_list_models(http):
    client = OllamaClient()
    models = client.list_models()
    assert [m.name for m in models] == ["ornith-1.5:9b", "nomic-embed-text:latest"]
    assert models[0].size == 6_600_000_000


def test_loaded_models(http):
    loaded = OllamaClient().loaded_models()
    assert loaded[0].name == "ornith-1.5:9b"
    assert loaded[0].size_vram == 7_100_000_000
    assert loaded[0].context_length == 100000


def test_context_length_discovery(http):
    client = OllamaClient(model="ornith-1.5:9b")
    assert client.loaded_context_length() == 100000
    assert client.model_context_length() == 262144


def test_chat_detailed_reports_usage(http):
    result = OllamaClient().chat_detailed([{"role": "user", "content": "hi"}])
    assert result["content"] == "chat reply"
    assert result["model"] == "ornith-1.5:9b"
    assert result["prompt_tokens"] == 17
    assert result["eval_tokens"] == 5
    assert result["done_reason"] == "stop"


def test_refresh_ok(http):
    status = OllamaClient(model="ornith-1.5:9b").refresh()
    assert status.server_ok
    assert status.current == "ornith-1.5:9b"
    assert status.current_loaded
    assert not status.error
    d = status.to_dict()
    assert d["current_loaded"] is True
    assert len(d["installed"]) == 2


def test_refresh_reports_missing_model(http):
    status = OllamaClient(model="does-not-exist").refresh()
    assert status.server_ok
    assert "not installed" in status.error


def test_refresh_server_down(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", boom)
    status = OllamaClient().refresh()
    assert status.server_ok is False
    assert "cannot reach Ollama" in status.error


def test_refresh_switches_model(http):
    client = OllamaClient(model="ornith-1.5:9b")
    status = client.refresh("nomic-embed-text:latest")
    assert client.model == "nomic-embed-text:latest"
    assert status.current == "nomic-embed-text:latest"


def test_require_model(http):
    client = OllamaClient()
    assert client.require_model("ornith-1.5:9b") == "ornith-1.5:9b"
    with pytest.raises(LLMError, match="not installed"):
        client.require_model("nope")


def test_generate_payload_and_thinking(http):
    client = OllamaClient(model="ornith-1.5:9b")
    out = client.generate("hi", system="sys", json_mode=True, temperature=0.0)
    assert out == "hello there"
    call = next(c for c in http if c["url"].endswith("/api/generate"))
    assert call["body"]["model"] == "ornith-1.5:9b"
    assert call["body"]["system"] == "sys"
    assert call["body"]["format"] == "json"
    assert call["body"]["stream"] is False


def test_generate_override():
    calls = []

    def fake(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "overridden"

    client = OllamaClient(generate=fake)
    assert client.generate("q") == "overridden"
    assert calls[0][0] == "q"


def test_generate_connection_error(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", boom)
    with pytest.raises(LLMError, match="Is `ollama serve` running"):
        OllamaClient().generate("hi")


def test_generate_http_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 404, "Not Found", None, io.BytesIO(b"model not found"),
        )

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", boom)
    with pytest.raises(LLMError, match="404"):
        OllamaClient().generate("hi")


def test_chat(http):
    out = OllamaClient().chat([{"role": "user", "content": "hi"}])
    assert out == "chat reply"


def test_embed(http):
    vectors = OllamaClient().embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    call = next(c for c in http if c["url"].endswith("/api/embed"))
    assert call["body"]["input"] == ["a", "b"]


def test_embed_empty():
    assert OllamaClient().embed([]) == []


def test_embed_falls_back_to_legacy_endpoint(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if req.full_url.endswith("/api/embed"):
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", None, io.BytesIO(b"no"),
            )
        return FakeResponse({"embedding": [1.0, 2.0]})

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    vectors = OllamaClient().embed(["x"])
    assert vectors == [[1.0, 2.0]]
    assert calls == [
        "http://127.0.0.1:11434/api/embed",
        "http://127.0.0.1:11434/api/embeddings",
    ]


def test_pull_success(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed("success"))
    assert OllamaClient().pull("m") == "success"


def test_pull_failure(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed("", "denied", 1))
    with pytest.raises(LLMError, match="denied"):
        OllamaClient().pull("m")
