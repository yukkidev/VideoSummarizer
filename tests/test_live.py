"""Live tests: require a running Ollama, or the network for the whisper sample.

Run with: pytest -m live
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from pipeline.llm import OllamaClient, client_from_config
from pipeline.models import Chunk, Video
from pipeline.summarizer import summarize_chunks
from pipeline.transcriber import faster_whisper_available, transcribe_audio

JFK_URL = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
JFK_TEXT = "ask not what your country can do for you"


@pytest.fixture(scope="module")
def jfk_wav(tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("whisper") / "jfk.wav"
    try:
        urllib.request.urlretrieve(JFK_URL, target)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot download sample audio: {exc}")
    return target


@pytest.mark.live
def test_live_faster_whisper_transcribes(jfk_wav: Path):
    if not faster_whisper_available():
        pytest.skip("faster-whisper not installed")
    transcript = transcribe_audio(
        jfk_wav, model="tiny", device="cpu", progress=lambda _m, _p: None,
    )
    assert transcript.segments
    assert JFK_TEXT in transcript.text.lower()


@pytest.mark.live
def test_live_ollama_refresh():
    client = OllamaClient()
    status = client.refresh()
    if not status.server_ok:
        pytest.skip("ollama is not running")
    assert status.installed
    assert status.current in {m.name for m in status.installed}


@pytest.mark.live
def test_live_summarize_short_transcript():
    llm = client_from_config()
    status = llm.refresh()
    if not status.server_ok:
        pytest.skip("ollama is not running")
    video = Video(
        video_id="live", url="https://example.com/live",
        title="JFK Inaugural Address", author="John F. Kennedy", duration=11,
    )
    chunks = [
        Chunk(
            index=0, start=0.0, end=11.0,
            text=(
                "And so, my fellow Americans: ask not what your country can do "
                "for you—ask what you can do for your country."
            ),
        )
    ]
    summary = summarize_chunks(chunks, video, llm)
    assert summary.model == status.current
    assert summary.summary
    assert len(summary.highlights) >= 3
